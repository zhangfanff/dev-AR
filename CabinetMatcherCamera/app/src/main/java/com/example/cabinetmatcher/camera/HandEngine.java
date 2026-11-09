package com.example.cabinetmatcher.camera;

import android.content.Context;
import android.graphics.Bitmap;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import java.util.List;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

/** All methods, including construction/reset/close, run on one persistent hand worker. */
final class HandEngine implements AutoCloseable {
    private final Context context;
    private HandLandmarker detector;
    HandEngine(Context context) { this.context=context.getApplicationContext(); reset(); }
    void reset() {
        close();
        BaseOptions base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(Delegate.CPU).build();
        detector=HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base).setRunningMode(RunningMode.VIDEO).setNumHands(2)
                .setMinHandDetectionConfidence(.5f).setMinHandPresenceConfidence(.5f).setMinTrackingConfidence(.5f).build());
    }
    float[][] detect(Bitmap bitmap, long timestampMs) {
        // Synchronous VIDEO mode retains temporal tracking while letting CameraX hold its
        // current ImageProxy until processing really finishes. No asynchronous frame queue.
        MPImage image=new BitmapImageBuilder(bitmap).build();
        try {
            HandLandmarkerResult result=detector.detectForVideo(image,timestampMs);
            float[][] hands=new float[result.landmarks().size()][HandSkeleton.LANDMARKS*3];
            for (int h=0;h<hands.length;h++) {
                List<NormalizedLandmark> points=result.landmarks().get(h);
                if (points.size()!=HandSkeleton.LANDMARKS) throw new IllegalStateException("Expected 21 hand landmarks");
                for (int i=0;i<points.size();i++) {
                    hands[h][3*i]=points.get(i).x(); hands[h][3*i+1]=points.get(i).y(); hands[h][3*i+2]=points.get(i).z();
                }
            }
            return hands;
        } finally { image.close(); }
    }
    @Override public void close() { if (detector!=null) { detector.close(); detector=null; } }
}
