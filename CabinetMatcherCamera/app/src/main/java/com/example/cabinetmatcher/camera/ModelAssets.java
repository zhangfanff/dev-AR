package com.example.cabinetmatcher.camera;

import android.content.Context;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Iterator;

final class ModelAssets {
    static File prepare(Context context) throws Exception {
        byte[] manifest;
        try (InputStream stream = context.getAssets().open("models_manifest.json")) { manifest = readAll(stream); }
        String revision = hex(MessageDigest.getInstance("SHA-256").digest(manifest)).substring(0, 16);
        File dir = new File(context.getFilesDir(), "models/" + revision);
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IllegalStateException("无法创建模型目录");
        JSONObject models = new JSONObject(new String(manifest, StandardCharsets.UTF_8)).getJSONObject("models");
        Iterator<String> names = models.keys();
        while (names.hasNext()) {
            String name = names.next();
            File target = new File(dir, name);
            String expected = models.getJSONObject(name).getString("sha256");
            if (target.isFile() && sha(target).equals(expected)) continue;
            copy(context, name, target);
            if (!sha(target).equals(expected)) throw new IllegalStateException("模型校验失败：" + name);
        }
        return dir;
    }
    private static void copy(Context context, String name, File target) throws Exception {
        File temporary = new File(target.getParentFile(), target.getName() + ".tmp");
        try (InputStream input = context.getAssets().open(name); FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[65536]; int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
        Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
    private static byte[] readAll(InputStream input) throws Exception {
        java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[8192]; int count;
        while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        return output.toByteArray();
    }
    private static String sha(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[65536]; int count;
            while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        return hex(digest.digest());
    }
    private static String hex(byte[] bytes) {
        StringBuilder hex = new StringBuilder();
        for (byte value : bytes) hex.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return hex.toString();
    }
}
