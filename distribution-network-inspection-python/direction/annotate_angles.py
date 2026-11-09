"""Visual manual direction labeling. No model predictions are shown by default."""

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
from wirelab.io import ROOT, manifest, read, write, sha, MaskStore, references

STATES = {
    "未检查": "unreviewed",
    "已有参考线": "labeled",
    "无可见导线": "no_wire",
    "无法判定": "ambiguous",
}
STATE_NAMES = {v: k for k, v in STATES.items()}
TAGS = [
    "近竖直",
    "斜向",
    "近水平",
    "多个出线方向",
    "植被遮挡",
    "建筑/车辆背景",
    "导线弯曲",
    "导线不清晰",
]
COLORS = ["#ffdf00", "#00ffff", "#ff66ff", "#58ff58", "#ff9944", "#88aaff"]
VIEWS = {
    "近垂直俯视，适用": "nadir",
    "斜视，不适用": "oblique",
    "视角不确定，暂不使用": "uncertain",
}
VIEW_NAMES = {v: k for k, v in VIEWS.items()}


class App:
    def __init__(self, window, root=ROOT, labels=None):
        self.window = window
        self.root = Path(root)
        self.items = manifest(root)["images"]
        self.manifest_hash = sha(self.root / "dataset/manifest.json")
        self.labels = Path(labels) if labels else self.root / "annotations"
        self.index = 0
        self.filtered = list(range(len(self.items)))
        self.points = []
        self.history = []
        self.dirty = False
        self.scale = 1.0
        self.doc = None
        self.store = None
        window.title("导线方向标注")
        window.geometry("1280x900")
        window.minsize(1050, 720)
        window.option_add("*Font", ("Microsoft YaHei UI", 10))
        outer = ttk.Frame(window, padding=8)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, width=220)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        self.filter = tk.StringVar(value="全部")
        self.active_filter = "全部"
        box = ttk.Combobox(
            left,
            textvariable=self.filter,
            values=["全部", "开发集", "测试集", "未完成", "视角剔除"],
            state="readonly",
        )
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", self.change_filter)
        self.progress = ttk.Label(left, text="", wraplength=210)
        self.progress.pack(fill="x", pady=6)
        self.listbox = tk.Listbox(
            left, exportselection=False, font=("Microsoft YaHei UI", 9)
        )
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.choose)
        ttk.Label(
            left,
            text="仅自采集配网图像。\n先看视角，斜视图可直接剔除。\n参考线不要照着算法输出画。",
            wraplength=210,
        ).pack(pady=8)
        center = ttk.Frame(outer)
        center.pack(side="left", fill="both", expand=True)
        tools = ttk.Frame(center)
        tools.pack(fill="x")
        for text, cmd in [
            ("适应窗口", self.fit_view),
            ("放大 +", lambda: self.zoom(1.3)),
            ("缩小 −", lambda: self.zoom(1 / 1.3)),
        ]:
            ttk.Button(tools, text=text, command=cmd).pack(side="left", padx=2)
        self.mask_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tools,
            text="辅助查看已有人工掩码",
            variable=self.mask_visible,
            command=self.render,
        ).pack(side="left", padx=5)
        frame = ttk.Frame(center)
        frame.pack(fill="both", expand=True, pady=5)
        self.canvas = tk.Canvas(frame, bg="#20242a", highlightthickness=0)
        sx = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        sy = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", lambda e: self.finish_line())
        self.canvas.bind(
            "<MouseWheel>", lambda e: self.zoom(1.15 if e.delta > 0 else 1 / 1.15)
        )
        self.hint = ttk.Label(center, text="", wraplength=700)
        self.hint.pack(fill="x")
        nav = ttk.Frame(center)
        nav.pack(fill="x", pady=5)
        ttk.Button(nav, text="上一张", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(nav, text="保存并确认 / 下一张", command=self.confirm_next).pack(
            side="left", padx=8
        )
        ttk.Button(nav, text="保存草稿", command=lambda: self.save(False)).pack(
            side="left"
        )
        right = ttk.Frame(outer, width=280)
        right.pack(side="left", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        self.info = ttk.Label(right, text="", wraplength=270)
        self.info.pack(fill="x", pady=(0, 10))
        ttk.Label(right, text="视角检查（斜视≠图中导线倾斜）").pack(anchor="w")
        self.viewpoint = tk.StringVar(value="近垂直俯视，适用")
        vb = ttk.Combobox(
            right, textvariable=self.viewpoint, values=list(VIEWS), state="readonly"
        )
        vb.pack(fill="x", pady=(0, 6))
        vb.bind("<<ComboboxSelected>>", lambda e: self.mark_dirty())
        self.region = tk.StringVar(value="upper")
        self.group = tk.IntVar(value=1)
        self.directed = tk.BooleanVar(value=True)
        ttk.Label(right, text="当前绘制区域").pack(anchor="w")
        rr = ttk.Frame(right)
        rr.pack(fill="x")
        ttk.Radiobutton(
            rr, text="上部 0–60%", variable=self.region, value="upper"
        ).pack(side="left")
        ttk.Radiobutton(
            rr, text="下部 40–100%", variable=self.region, value="lower"
        ).pack(side="left")
        gg = ttk.Frame(right)
        gg.pack(fill="x", pady=6)
        ttk.Label(gg, text="出线方向组号").pack(side="left")
        ttk.Spinbox(gg, from_=1, to=99, textvariable=self.group, width=4).pack(
            side="left", padx=5
        )
        ttk.Button(gg, text="新组", command=self.next_group).pack(side="left")
        ttk.Checkbutton(right, text="箭头方向也能确定", variable=self.directed).pack(
            anchor="w"
        )
        ttk.Label(
            right,
            text="沿导线画线：首点靠近杆塔，末点朝外。\n同一出线方向使用相同组号。\n仅能确定线轴时取消“箭头方向也能确定”。",
            wraplength=270,
        ).pack(fill="x", pady=7)
        ttk.Button(
            right, text="完成当前线（右键 / Enter）", command=self.finish_line
        ).pack(fill="x")
        ttk.Button(right, text="撤销（Ctrl+Z）", command=self.undo).pack(
            fill="x", pady=3
        )
        ttk.Button(right, text="删除选中参考线", command=self.delete_line).pack(
            fill="x"
        )
        self.lines = tk.Listbox(
            right, height=7, exportselection=False, font=("Microsoft YaHei UI", 9)
        )
        self.lines.pack(fill="x", pady=6)
        self.lines.bind("<<ListboxSelect>>", lambda e: self.render())
        self.state_vars = {}
        ttk.Label(right, text="两个区域都需检查").pack(anchor="w")
        for region, title in [("upper", "上部"), ("lower", "下部")]:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=title, width=5).pack(side="left")
            v = tk.StringVar(value="未检查")
            self.state_vars[region] = v
            cb = ttk.Combobox(
                row, textvariable=v, values=list(STATES), state="readonly", width=16
            )
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", lambda e: self.mark_dirty())
        self.tags = {t: tk.BooleanVar(value=False) for t in TAGS}
        tagbox = ttk.LabelFrame(right, text="场景标签（人工确认）")
        tagbox.pack(fill="x", pady=6)
        for i, (t, v) in enumerate(self.tags.items()):
            ttk.Checkbutton(tagbox, text=t, variable=v, command=self.mark_dirty).grid(
                row=i // 2, column=i % 2, sticky="w"
            )
        ttk.Label(right, text="备注（可选）").pack(anchor="w")
        self.notes = tk.Text(right, height=3, wrap="word")
        self.notes.pack(fill="x")
        self.notes.bind("<KeyRelease>", lambda e: self.mark_dirty())
        window.bind("<Return>", lambda e: self.finish_line())
        window.bind("<Control-z>", lambda e: self.undo())
        window.bind("<Escape>", lambda e: self.cancel_points())
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_list()
        self.load(0)
        window.after(150, self.fit_view)

    def path(self, item):
        return self.labels / (item["key"] + ".json")

    def mark_dirty(self):
        self.dirty = True

    def refresh_list(self):
        self.listbox.delete(0, "end")
        done = 0
        for index in self.filtered:
            item = self.items[index]
            p = self.path(item)
            ok = p.exists() and read(p).get("reviewed", False)
            done += ok
            text = (
                ("✓ " if ok else "○ ")
                + ("测 " if item["role"] == "test" else "开 ")
                + item["key"]
                + (" 本地" if item["source"] == "local" else "")
            )
            self.listbox.insert("end", text)
        dev = sum(x["role"] == "dev" for x in self.items)
        test = sum(x["role"] == "test" for x in self.items)
        self.progress.configure(
            text=f"当前筛选 {done}/{len(self.filtered)} 已确认\n本地 {len(self.items)} 张；开发{dev} / 测试{test}"
        )
        if self.index in self.filtered:
            i = self.filtered.index(self.index)
            self.listbox.selection_set(i)
            self.listbox.see(i)

    def change_filter(self, event=None):
        if not self.save_if_dirty():
            self.filter.set(self.active_filter)
            return
        name = self.filter.get()
        self.active_filter = name
        self.filtered = []
        for i, x in enumerate(self.items):
            p = self.path(x)
            done = p.exists() and read(p).get("reviewed", False)
            excluded = p.exists() and read(p).get("viewpoint", "nadir") in (
                "oblique",
                "uncertain",
            )
            if (
                name == "全部"
                or (name == "开发集" and x["role"] == "dev")
                or (name == "测试集" and x["role"] == "test")
                or (name == "未完成" and not done)
                or (name == "视角剔除" and excluded)
            ):
                self.filtered.append(i)
        self.refresh_list()
        if self.filtered:
            self.load(self.filtered[0])

    def choose(self, event=None):
        chosen = self.listbox.curselection()
        if chosen:
            index = self.filtered[chosen[0]]
            if index != self.index:
                if self.save_if_dirty():
                    self.load(index)
                else:
                    self.refresh_list()

    def load(self, index):
        self.index = index
        self.item = self.items[index]
        if sha(self.root / self.item["image_path"]) != self.item["image_sha256"]:
            raise RuntimeError("Image was modified.")
        self.original = Image.open(self.root / self.item["image_path"]).convert("RGB")
        p = self.path(self.item)
        self.doc = (
            read(p)
            if p.exists()
            else {
                "schema": "wire_direction_reference_v1",
                "key": self.item["key"],
                "image_sha256": self.item["image_sha256"],
                "reviewed": False,
                "regions": {"upper": "unreviewed", "lower": "unreviewed"},
                "lines": [],
                "tags": [],
                "notes": "",
            }
        )
        self.points = []
        self.history = []
        self.dirty = False
        self.region.set("upper")
        self.group.set(1)
        self.viewpoint.set(VIEW_NAMES[self.doc.get("viewpoint", "nadir")])
        for r, v in self.state_vars.items():
            v.set(STATE_NAMES[self.doc["regions"][r]])
        for t, v in self.tags.items():
            v.set(t in self.doc.get("tags", []))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", self.doc.get("notes", ""))
        pitch = self.item.get("camera_metadata", {}).get("GimbalPitchDegree")
        pose = f"\n云台俯仰 {pitch:+.1f}°" if pitch is not None else ""
        seen = (
            "\n原训练图，仅用于开发调试"
            if self.item.get("segmentation_training_seen")
            else ""
        )
        self.info.configure(
            text=f"{self.index + 1}/{len(self.items)}  {self.item['key']}\n"
            + ("测试集" if self.item["role"] == "test" else "开发集")
            + " | 自采集\n"
            + self.item["file_name"]
            + pose
            + seen
        )
        self.refresh_lines()
        self.render()
        self.refresh_list()

    def fit_view(self):
        self.window.update_idletasks()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 30 or height < 30:
            self.window.after(200, self.fit_view)
            return
        self.scale = min(
            (width - 10) / self.original.width, (height - 10) / self.original.height
        )
        self.render()

    def zoom(self, multiplier):
        self.scale = max(0.25, min(6.0, self.scale * multiplier))
        self.render()

    def render(self):
        if self.doc is None:
            return
        image = self.original.copy()
        if self.mask_visible.get():
            if not self.item.get("has_gt_mask", True):
                self.mask_visible.set(False)
                messagebox.showinfo(
                    "没有分割掩码",
                    "这张现场照片还没有人工分割掩码，请直接在图像上标方向线。不会显示模型预测作为参考。",
                )
                self.render()
                return
            if self.store is None:
                self.store = MaskStore(self.root)
            pixels = np.asarray(image).astype(float)
            for i, (mask, _) in enumerate(self.store.load(self.item, "gt")):
                color = (
                    np.array([0, 220, 255]) if i % 2 == 0 else np.array([255, 0, 220])
                )
                pixels[mask] = 0.65 * pixels[mask] + 0.35 * color
            image = Image.fromarray(pixels.astype(np.uint8))
        w = max(1, int(image.width * self.scale))
        h = max(1, int(image.height * self.scale))
        self.photo = ImageTk.PhotoImage(image.resize((w, h), Image.Resampling.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, w, h))
        for ratio, text in [(0.4, "下部起点 40%"), (0.6, "上部终点 60%")]:
            y = h * ratio
            self.canvas.create_line(0, y, w, y, fill="#ffffff", dash=(6, 5))
            self.canvas.create_text(5, y + 3, text=text, fill="white", anchor="nw")
        for i, line in enumerate(self.doc["lines"]):
            xy = [z * self.scale for point in line["points"] for z in point]
            c = COLORS[(line["group"] - 1) % len(COLORS)]
            selected = i in self.lines.curselection()
            self.canvas.create_line(
                *xy,
                fill=c,
                width=4 if selected else 2,
                arrow="last" if line.get("directed") else "none",
            )
            self.canvas.create_text(
                xy[0] + 5,
                xy[1] - 5,
                text=f"{i + 1} / {line['region']} G{line['group']}",
                fill=c,
                anchor="sw",
            )
        if self.points:
            xy = [z * self.scale for point in self.points for z in point]
            if len(self.points) > 1:
                self.canvas.create_line(*xy, fill="white", width=2, dash=(4, 3))
            for x, y in self.points:
                self.canvas.create_oval(
                    x * self.scale - 3,
                    y * self.scale - 3,
                    x * self.scale + 3,
                    y * self.scale + 3,
                    fill="white",
                )
        self.hint.configure(
            text=f"缩放 {self.scale:.0%} | 左键加点，右键/Enter完成线，Esc取消当前线；上/下区域重叠20%。"
        )

    def click(self, event):
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        if not (0 <= x < self.item["width"] and 0 <= y < self.item["height"]):
            return
        region = self.region.get()
        h = self.item["height"]
        if (region == "upper" and y >= 0.6 * h) or (region == "lower" and y < 0.4 * h):
            self.hint.configure(text="点不在当前区域内，请切换上部/下部。")
            return
        self.points.append([round(x, 3), round(y, 3)])
        self.render()

    def finish_line(self):
        if len(self.points) < 2:
            return
        try:
            group = int(self.group.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("组号无效", "请输入1到99之间的整数组号。")
            return
        if not 1 <= group <= 99:
            messagebox.showwarning("组号无效", "请输入1到99之间的整数组号。")
            return
        region = self.region.get()
        if any(
            (region == "upper" and p[1] >= 0.6 * self.item["height"])
            or (region == "lower" and p[1] < 0.4 * self.item["height"])
            for p in self.points
        ):
            messagebox.showwarning(
                "区域不一致", "当前线包含其他区域的点。请切回原区域完成，或按Esc重画。"
            )
            return
        if np.linalg.norm(np.diff(np.asarray(self.points), axis=0), axis=1).sum() < 20:
            messagebox.showwarning(
                "参考线过短", "请尽量选择较长的可辨认导线片段，至少20像素。"
            )
            return
        self.history.append(copy.deepcopy(self.doc))
        self.doc["lines"].append(
            {
                "region": region,
                "group": group,
                "points": self.points.copy(),
                "directed": self.directed.get(),
            }
        )
        self.state_vars[region].set("已有参考线")
        self.doc["regions"][region] = "labeled"
        self.points = []
        self.dirty = True
        self.refresh_lines()
        self.render()

    def cancel_points(self):
        self.points = []
        self.render()

    def next_group(self):
        groups = [
            l["group"] for l in self.doc["lines"] if l["region"] == self.region.get()
        ]
        self.group.set(max(groups, default=0) + 1)

    def refresh_lines(self):
        self.lines.delete(0, "end")
        for i, line in enumerate(self.doc["lines"]):
            angle = references({"lines": [line]}, line["region"])[0]["angle"]
            self.lines.insert(
                "end", f"{i + 1}: {line['region']} G{line['group']}  {angle:+.1f}°"
            )

    def delete_line(self):
        selected = self.lines.curselection()
        if not selected:
            return
        self.history.append(copy.deepcopy(self.doc))
        self.doc["lines"].pop(selected[0])
        self.dirty = True
        self.refresh_lines()
        self.render()

    def undo(self):
        if self.points:
            self.points.pop()
        elif self.history:
            self.doc = self.history.pop()
            self.dirty = True
            for r, v in self.state_vars.items():
                v.set(STATE_NAMES[self.doc["regions"][r]])
            self.refresh_lines()
        self.render()

    def save(self, reviewed):
        if sha(self.root / "dataset/manifest.json") != self.manifest_hash:
            messagebox.showwarning(
                "样本清单已更新",
                "样本已重新整理。请重新打开标注工具，以免用旧清单继续写入。当前窗口中的未保存参考线仍保留。",
            )
            return False
        if self.points:
            messagebox.showwarning("有未完成的线", "请先右键完成参考线，或按Esc取消。")
            return False
        regions = {r: STATES[v.get()] for r, v in self.state_vars.items()}
        view = VIEWS[self.viewpoint.get()]
        if reviewed and view == "nadir":
            for r, state in regions.items():
                count = sum(l["region"] == r for l in self.doc["lines"])
                if (
                    state == "unreviewed"
                    or (state == "labeled" and not count)
                    or (state != "labeled" and count)
                ):
                    messagebox.showwarning(
                        "区域状态需要确认",
                        f"{STATE_NAMES[state]}：{r}区域的状态与参考线不一致。\n有线选“已有参考线”；没法判断选“无法判定”。",
                    )
                    return False
        self.doc.update(
            regions=regions,
            reviewed=reviewed,
            viewpoint=view,
            tags=[t for t, v in self.tags.items() if v.get()],
            notes=self.notes.get("1.0", "end").strip(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        target = self.path(self.item)
        if target.exists() and read(target) != self.doc:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            write(
                self.labels / "backups" / (self.item["key"] + "_" + stamp + ".json"),
                read(target),
            )
        write(target, self.doc)
        self.dirty = False
        self.refresh_list()
        return True

    def save_if_dirty(self):
        if self.points:
            return self.save(False)
        if self.dirty:
            return self.save(False)
        return True

    def move(self, step):
        if not self.save_if_dirty():
            return
        if self.index not in self.filtered:
            return
        i = self.filtered.index(self.index) + step
        if 0 <= i < len(self.filtered):
            self.load(self.filtered[i])

    def confirm_next(self):
        if self.save(True):
            self.move(1)

    def close(self):
        if self.points:
            if not messagebox.askyesno(
                "退出", "当前线未完成，退出时放弃这条未完成的线？"
            ):
                return
            self.points = []
        if self.save_if_dirty():
            self.window.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--screenshot",
        type=Path,
        help="Visual QA only: capture the clean window and exit; do not save labels.",
    )
    args = p.parse_args()
    window = tk.Tk()
    app = App(window, args.root)
    if args.screenshot:

        def capture():
            # Capture only this process's own Tk window, including when occluded.
            # Never capture unrelated desktop applications.
            import ctypes
            from ctypes import wintypes

            app.fit_view()
            window.update_idletasks()
            user = ctypes.windll.user32
            gdi = ctypes.windll.gdi32
            user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user.GetAncestor.restype = wintypes.HWND
            user.GetDC.argtypes = [wintypes.HWND]
            user.GetDC.restype = wintypes.HDC
            gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
            gdi.CreateCompatibleDC.restype = wintypes.HDC
            gdi.CreateCompatibleBitmap.argtypes = [
                wintypes.HDC,
                ctypes.c_int,
                ctypes.c_int,
            ]
            gdi.CreateCompatibleBitmap.restype = wintypes.HBITMAP
            gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
            gdi.SelectObject.restype = wintypes.HGDIOBJ
            user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
            hwnd = user.GetAncestor(window.winfo_id(), 2)
            rect = wintypes.RECT()
            user.GetClientRect(hwnd, ctypes.byref(rect))
            w, h = rect.right, rect.bottom
            dc = user.GetDC(hwnd)
            memory = gdi.CreateCompatibleDC(dc)
            bitmap = gdi.CreateCompatibleBitmap(dc, w, h)
            old = gdi.SelectObject(memory, bitmap)
            if not user.PrintWindow(hwnd, memory, 3):
                raise RuntimeError("Own-window capture failed.")

            class Header(ctypes.Structure):
                _fields_ = [
                    ("size", wintypes.DWORD),
                    ("width", wintypes.LONG),
                    ("height", wintypes.LONG),
                    ("planes", wintypes.WORD),
                    ("bits", wintypes.WORD),
                    ("compression", wintypes.DWORD),
                    ("image_size", wintypes.DWORD),
                    ("xppm", wintypes.LONG),
                    ("yppm", wintypes.LONG),
                    ("colors", wintypes.DWORD),
                    ("important", wintypes.DWORD),
                ]

            header = Header(ctypes.sizeof(Header), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
            pixels = ctypes.create_string_buffer(w * h * 4)
            gdi.GetDIBits.argtypes = [
                wintypes.HDC,
                wintypes.HBITMAP,
                wintypes.UINT,
                wintypes.UINT,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.UINT,
            ]
            if not gdi.GetDIBits(memory, bitmap, 0, h, pixels, ctypes.byref(header), 0):
                raise RuntimeError("Own-window bitmap read failed.")
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.frombuffer("RGB", (w, h), pixels, "raw", "BGRX", 0, 1).save(
                args.screenshot
            )
            gdi.SelectObject(memory, old)
            gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
            gdi.DeleteObject(bitmap)
            gdi.DeleteDC.argtypes = [wintypes.HDC]
            gdi.DeleteDC(memory)
            user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
            user.ReleaseDC(hwnd, dc)
            window.destroy()

        window.after(1800, capture)
    window.mainloop()


if __name__ == "__main__":
    main()
