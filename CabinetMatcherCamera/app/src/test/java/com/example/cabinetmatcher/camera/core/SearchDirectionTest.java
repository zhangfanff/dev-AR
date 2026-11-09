package com.example.cabinetmatcher.camera.core;

import org.junit.Test;
import java.util.Arrays;
import java.util.Collections;
import static org.junit.Assert.*;

public class SearchDirectionTest {
    private Detection point(int id,double x,double y,double confidence) {
        return new Detection(id,confidence,new double[]{x-1,y-1,x+1,y+1},null,null);
    }
    private MatchReport match(SearchDirection direction,double angle,double lateral,Detection... texts) {
        return GreedyMatcher.match(500,500,Collections.singletonList(point(0,100,100,.9)),Arrays.asList(texts),100,lateral,angle,direction);
    }
    @Test public void eachCardinalDirectionSelectsOnlyItsOwnLabel() {
        Detection[] texts={point(0,100,140,0),point(1,100,60,0),point(2,60,100,0),point(3,140,100,0)};
        for(SearchDirection direction:new SearchDirection[]{SearchDirection.DOWN,SearchDirection.UP,SearchDirection.LEFT,SearchDirection.RIGHT}) {
            MatchReport result=match(direction,30,Double.NaN,texts);
            assertEquals(1,result.matches.size()); assertEquals(direction.ordinal(),result.matches.get(0).text.id);
            assertEquals(direction,result.searchDirection);
        }
    }
    @Test public void allDirectionsUseDistanceAcrossAxesInsteadOfAxisOrder() {
        MatchReport result=match(SearchDirection.ALL,30,Double.NaN,point(0,100,160,0),point(1,100,80,0),point(2,70,100,0),point(3,140,100,0));
        assertEquals(1,result.matches.size()); assertEquals(1,result.matches.get(0).text.id);
        assertEquals(SearchDirection.UP,SearchDirection.ALL.axis(0,-20));
    }
    @Test public void fourWayUnionStillAssignsOneTextToOnlyOneConfidenceRankedKnob() {
        Detection high=point(1,100,100,.95),low=point(0,100,115,.6),text=point(0,100,140,0);
        MatchReport result=GreedyMatcher.match(500,500,Arrays.asList(low,high),Collections.singletonList(text),100,Double.NaN,30,SearchDirection.ALL);
        assertEquals(1,result.matches.size()); assertEquals(high.id,result.matches.get(0).knob.id);
        assertEquals("eligible_texts_already_used",result.decisions.get(1).reason);
    }
    @Test public void allSectorBoundariesAreInclusiveButSlightlyOutsideIsRejected() {
        double side=30*Math.tan(Math.toRadians(30));
        double[][] offsets={{side,30},{side,-30},{-30,side},{30,side}};
        for(int i=0;i<4;i++) {
            SearchDirection direction=SearchDirection.values()[i]; double dx=offsets[i][0],dy=offsets[i][1];
            assertEquals(1,match(direction,30,Double.NaN,point(0,100+dx,100+dy,0)).matches.size());
            if(i<2) dx+=.01; else dy+=.01;
            assertEquals(0,match(direction,30,Double.NaN,point(0,100+dx,100+dy,0)).matches.size());
        }
    }
    @Test public void sidewaysSearchRotatesOptionalLateralLimit() {
        assertEquals(1,match(SearchDirection.RIGHT,45,5,point(0,140,104,0)).matches.size());
        assertEquals(0,match(SearchDirection.RIGHT,45,5,point(0,140,106,0)).matches.size());
        assertEquals(1,match(SearchDirection.UP,45,5,point(0,104,60,0)).matches.size());
    }
    @Test public void narrowFourWaySectorsLeaveDiagonalGapsAndRejectCoincidentCenters() {
        assertEquals(0,match(SearchDirection.ALL,20,Double.NaN,point(0,130,130,0)).matches.size());
        assertEquals(1,match(SearchDirection.ALL,45,Double.NaN,point(0,130,130,0)).matches.size());
        assertEquals(0,match(SearchDirection.ALL,45,Double.NaN,point(0,100,100,0)).matches.size());
    }
    @Test public void defaultsEnableTextOrientationAndPreserveDownwardSearch() {
        CabinetPipeline.Settings settings=new CabinetPipeline.Settings();
        assertTrue(settings.angleClassifier); assertEquals(SearchDirection.DOWN,settings.searchDirection); settings.validate();
    }
    @Test(expected=IllegalArgumentException.class) public void nullDirectionCannotRun() {
        match(null,45,Double.NaN,point(0,100,140,0));
    }
}
