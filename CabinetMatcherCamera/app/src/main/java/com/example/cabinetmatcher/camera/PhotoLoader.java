package com.example.cabinetmatcher.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import androidx.exifinterface.media.ExifInterface;
import java.io.InputStream;
import com.example.cabinetmatcher.camera.core.TimingReport;

final class PhotoLoader {
    interface StreamSource { InputStream open() throws Exception; }
    static final class Photo {
        final Bitmap bitmap;
        final int sourceWidth, sourceHeight;
        final TimingReport timing;
        Photo(Bitmap bitmap, int sourceWidth, int sourceHeight, TimingReport timing) {
            this.bitmap = bitmap; this.sourceWidth = sourceWidth; this.sourceHeight = sourceHeight;
            this.timing = timing;
        }
    }
    static Photo load(Context context, Uri uri) throws Exception {
        return decode(() -> {
            InputStream input = context.getContentResolver().openInputStream(uri);
            if (input == null) throw new IllegalArgumentException("无法读取所选图片");
            return input;
        });
    }
    static Photo demo(Context context) throws Exception { return decode(() -> context.getAssets().open("demo_cabinet.jpg")); }
    private static Photo decode(StreamSource source) throws Exception {
        long totalStart = System.nanoTime(), started = totalStart;
        TimingReport timing = new TimingReport();
        BitmapFactory.Options bounds = new BitmapFactory.Options(); bounds.inJustDecodeBounds = true;
        try (InputStream input = source.open()) { BitmapFactory.decodeStream(input, null, bounds); }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw new IllegalArgumentException("无法解码这张图片");
        timing.record("image_load.bounds", started);
        started = System.nanoTime();
        int orientation;
        try (InputStream input = source.open()) {
            orientation = new ExifInterface(input).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        }
        timing.record("image_load.exif", started);
        started = System.nanoTime();
        int sample = 1;
        while (Math.max(bounds.outWidth, bounds.outHeight) / sample > 5120) sample *= 2;
        BitmapFactory.Options options = new BitmapFactory.Options(); options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap;
        try (InputStream input = source.open()) { bitmap = BitmapFactory.decodeStream(input, null, options); }
        if (bitmap == null) throw new IllegalArgumentException("无法解码图片");
        timing.record("image_load.decode", started);
        started = System.nanoTime();
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: matrix.setScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_180: matrix.setRotate(180); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: matrix.setScale(1, -1); break;
            case ExifInterface.ORIENTATION_TRANSPOSE: matrix.setRotate(90); matrix.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_90: matrix.setRotate(90); break;
            case ExifInterface.ORIENTATION_TRANSVERSE: matrix.setRotate(-90); matrix.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_270: matrix.setRotate(-90); break;
        }
        boolean swap = orientation == ExifInterface.ORIENTATION_TRANSPOSE || orientation == ExifInterface.ORIENTATION_TRANSVERSE ||
                orientation == ExifInterface.ORIENTATION_ROTATE_90 || orientation == ExifInterface.ORIENTATION_ROTATE_270;
        int originalW = swap ? bounds.outHeight : bounds.outWidth, originalH = swap ? bounds.outWidth : bounds.outHeight;
        try {
            if (!matrix.isIdentity()) {
                Bitmap upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                if (upright != bitmap) { bitmap.recycle(); bitmap = upright; }
            }
            timing.record("image_load.orientation", started);
            started = System.nanoTime();
            double ratio = Math.min(1.0, 2560.0 / Math.max(bitmap.getWidth(), bitmap.getHeight()));
            if (ratio < 1) {
                Bitmap smaller = Bitmap.createScaledBitmap(bitmap, Math.max(1, (int) Math.round(bitmap.getWidth() * ratio)),
                        Math.max(1, (int) Math.round(bitmap.getHeight() * ratio)), true);
                if (smaller != bitmap) { bitmap.recycle(); bitmap = smaller; }
            }
            timing.record("image_load.resize", started);
            timing.record("image_load.total", totalStart);
            return new Photo(bitmap, originalW, originalH, timing);
        } catch (RuntimeException error) { bitmap.recycle(); throw error; }
    }
}
