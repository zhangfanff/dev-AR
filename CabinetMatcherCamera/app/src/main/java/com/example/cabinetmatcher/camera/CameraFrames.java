package com.example.cabinetmatcher.camera;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import androidx.camera.core.ImageProxy;
import com.example.cabinetmatcher.camera.core.YuvPacking;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

final class CameraFrames {
    static Mat bgr(ImageProxy frame) {
        if (frame.getFormat() != ImageFormat.YUV_420_888) throw new IllegalArgumentException("需要 YUV 视频帧");
        ImageProxy.PlaneProxy[] p = frame.getPlanes();
        if (p.length != 3) throw new IllegalArgumentException("视频帧平面数量无效");
        byte[] pixels = YuvPacking.nv21(frame.getWidth(), frame.getHeight(),
                p[0].getBuffer(), p[0].getRowStride(), p[0].getPixelStride(),
                p[1].getBuffer(), p[1].getRowStride(), p[1].getPixelStride(),
                p[2].getBuffer(), p[2].getRowStride(), p[2].getPixelStride());
        Mat yuv = new Mat(frame.getHeight() * 3 / 2, frame.getWidth(), CvType.CV_8UC1);
        Mat full = new Mat(), output = new Mat();
        try {
            yuv.put(0, 0, pixels); Imgproc.cvtColor(yuv, full, Imgproc.COLOR_YUV2BGR_NV21);
            Rect area = frame.getCropRect();
            if (area.left < 0 || area.top < 0 || area.right > full.cols() || area.bottom > full.rows()
                    || area.width() <= 0 || area.height() <= 0) throw new IllegalArgumentException("视频帧裁剪范围无效");
            Mat crop = full.submat(area.top, area.bottom, area.left, area.right);
            try { crop.copyTo(output); } finally { crop.release(); }
            rotate(output, frame.getImageInfo().getRotationDegrees());
            return output;
        } catch (RuntimeException error) { output.release(); throw error; }
        finally { yuv.release(); full.release(); }
    }
    static void rotate(Mat image, int degrees) {
        switch (degrees) {
            case 0: break;
            case 90: Core.rotate(image, image, Core.ROTATE_90_CLOCKWISE); break;
            case 180: Core.rotate(image, image, Core.ROTATE_180); break;
            case 270: Core.rotate(image, image, Core.ROTATE_90_COUNTERCLOCKWISE); break;
            default: throw new IllegalArgumentException("视频帧方向必须为 0/90/180/270 度");
        }
    }
    static Bitmap bitmap(Mat bgr) {
        Mat rgba = new Mat();
        try {
            Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA);
            Bitmap bitmap = Bitmap.createBitmap(bgr.cols(), bgr.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, bitmap); return bitmap;
        } finally { rgba.release(); }
    }
}
