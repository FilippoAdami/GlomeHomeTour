package com.glomehometour.arscan

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class TrajectoryCsvTest {
    @Test
    fun `only tracking poses contribute to the camera path`() {
        val csv = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported\r\n" +
            "1,1,2,3,0,0,0,1,TRACKING,1\r\n" +
            "2,100,200,300,0,0,0,1,PAUSED,0\r\n" +
            "3,NaN,0,0,0,0,0,1,TRACKING,0\r\n"

        assertArrayEquals(floatArrayOf(1f, 2f, 3f), trajectoryPointsFromCsv(csv), 0f)
    }
}
