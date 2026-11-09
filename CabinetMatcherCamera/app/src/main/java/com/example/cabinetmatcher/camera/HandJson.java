package com.example.cabinetmatcher.camera;

import org.json.JSONArray;
import org.json.JSONObject;

final class HandJson {
    static JSONObject write(float[][] hands,int width,int height,boolean completed,long frameId,long cameraNs,long modelTimestampMs) throws Exception {
        JSONArray values=new JSONArray();
        for(int h=0;h<hands.length;h++) {
            float[] pixels=HandSkeleton.pixels(hands[h],width,height); JSONArray points=new JSONArray();
            for(int i=0;i<21;i++) points.put(new JSONObject().put("id",i)
                    .put("x_normalized",hands[h][3*i]).put("y_normalized",hands[h][3*i+1]).put("z_normalized",hands[h][3*i+2])
                    .put("x_px",pixels[2*i]).put("y_px",pixels[2*i+1]));
            values.put(new JSONObject().put("hand_index",h).put("landmarks",points));
        }
        return new JSONObject().put("completed",completed).put("count",hands.length).put("frame_id",frameId)
                .put("camera_timestamp_ns",cameraNs).put("mediapipe_timestamp_ms",modelTimestampMs)
                .put("same_frame_as_cabinet",completed).put("mode","21_point_hand_landmarks")
                .put("coordinate_system","upright analysis frame pixels; normalized z is model-relative depth, not meters")
                .put("hands",values);
    }
}
