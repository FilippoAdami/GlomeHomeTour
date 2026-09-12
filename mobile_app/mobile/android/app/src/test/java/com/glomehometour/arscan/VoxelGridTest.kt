package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoxelGridTest {

    @Test
    fun `key packing round-trips negative coordinates`() {
        for (c in listOf(0, 1, -1, 1000, -1000, 1 shl 19, -(1 shl 19))) {
            val key = VoxelGrid.pack(c, -c, c / 2)
            assertEquals(c, VoxelGrid.unpackX(key))
            assertEquals(-c, VoxelGrid.unpackY(key))
            assertEquals(c / 2, VoxelGrid.unpackZ(key))
        }
    }

    @Test
    fun `ray marks the hit occupied, the space before it free, and behind it occluded`() {
        val grid = VoxelGrid()
        // Camera at the origin looking down +Z at a surface 2 m away.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)

        assertEquals(VoxelGrid.OCCUPIED, grid.stateAt(0f, 0f, 2f))
        assertEquals(VoxelGrid.FREE, grid.stateAt(0f, 0f, 1f))
        assertEquals(VoxelGrid.FREE, grid.stateAt(0f, 0f, 0.5f))
        // Behind the surface, within the occlusion depth cap: actively hidden, not unknown.
        assertEquals(VoxelGrid.OCCLUDED, grid.stateAt(0f, 0f, 2.5f))
        // Past the occlusion depth cap: genuinely unmapped, not carved either way.
        assertEquals(VoxelGrid.UNSET, grid.stateAt(0f, 0f, 2f + VoxelGrid.OCCLUSION_DEPTH_M + 0.5f))
        // ...and neither does anything off the ray.
        assertEquals(VoxelGrid.UNSET, grid.stateAt(1f, 0f, 1f))
    }

    @Test
    fun `occluded voxel is promoted to free once a later ray resolves it`() {
        val grid = VoxelGrid()
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        assertEquals(VoxelGrid.OCCLUDED, grid.stateAt(0f, 0f, 2.5f))
        // The occlusion cone behind the hit is a full metre deep, sampled well below the 10 cm
        // voxel size, so it spans several distinct voxels -- not just the one probed above.
        assertTrue(grid.occludedCount > 1)

        // A second, longer ray straight through the same spot resolves it as passable.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 4f)
        assertEquals(VoxelGrid.FREE, grid.stateAt(0f, 0f, 2.5f))
        assertTrue(grid.occludedCount > 0) // only the voxels this second ray actually crossed clear
    }

    @Test
    fun `occluded never overwrites an already-resolved voxel`() {
        val grid = VoxelGrid()
        // First: a long, clear ray marks z=2.5 FREE, on its way to a hit at z=5.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 5f)
        assertEquals(VoxelGrid.FREE, grid.stateAt(0f, 0f, 2.5f))

        // Then: a second, shorter ray hits at z=2, whose occlusion cone (2 to 3 m) covers the
        // same voxel -- occlusion must not clobber a state a ray has already resolved.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        assertEquals(VoxelGrid.FREE, grid.stateAt(0f, 0f, 2.5f))
    }

    @Test
    fun `free never overwrites an occupied voxel`() {
        val grid = VoxelGrid()
        grid.integrate(0f, 0f, 0f, 0f, 0f, 1f)
        assertEquals(VoxelGrid.OCCUPIED, grid.stateAt(0f, 0f, 1f))
        // A longer ray through the same spot: its carve passes right over that voxel.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 3f)
        assertEquals(VoxelGrid.OCCUPIED, grid.stateAt(0f, 0f, 1f))
        assertEquals(2, grid.occupiedCount)
    }

    @Test
    fun `two views become parallax-verified only past 25 degrees`() {
        val narrow = VoxelGrid()
        // Same surface point seen from two cameras 20 degrees apart about it.
        narrow.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        narrow.integrate(sinDeg(20f) * 2f, 0f, 2f - cosDeg(20f) * 2f, 0f, 0f, 2f)
        assertEquals(1, narrow.occupiedCount)
        assertEquals(0, narrow.verifiedCount)
        assertTrue(narrow.parallaxDegAt(0f, 0f, 2f) > 19f)

        val wide = VoxelGrid()
        wide.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        wide.integrate(sinDeg(30f) * 2f, 0f, 2f - cosDeg(30f) * 2f, 0f, 0f, 2f)
        assertEquals(1, wide.occupiedCount)
        assertEquals(1, wide.verifiedCount)
        assertEquals(1f, wide.coverageFraction(), 1e-6f)
    }

    @Test
    fun `parallax record keeps the widest angle, not the latest`() {
        val grid = VoxelGrid()
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        grid.integrate(sinDeg(40f) * 2f, 0f, 2f - cosDeg(40f) * 2f, 0f, 0f, 2f)
        grid.integrate(0.01f, 0f, 0f, 0f, 0f, 2f) // back to almost the first viewpoint
        assertTrue(grid.parallaxDegAt(0f, 0f, 2f) > 39f)
        assertEquals(1, grid.verifiedCount)
    }

    @Test
    fun `samples outside the trusted depth range are ignored`() {
        val grid = VoxelGrid()
        grid.integrate(0f, 0f, 0f, 0f, 0f, 0.1f) // closer than MIN_RANGE_M
        grid.integrate(0f, 0f, 0f, 0f, 0f, 40f) // further than MAX_RANGE_M
        assertEquals(0, grid.occupiedCount)
        assertEquals(0, grid.freeCount)
    }

    @Test
    fun `frontier target points into the unmapped side of the carved volume`() {
        val grid = VoxelGrid()
        // A wall at z = +3, seen from the origin: everything between is carved free, so the only
        // unknown space adjacent to free space is off to the sides.
        var x = -1f
        while (x <= 1f) {
            grid.integrate(0f, 0f, 0f, x, 0f, 3f)
            x += 0.05f
        }
        val target = FloatArray(3)
        assertTrue(grid.frontierTarget(0f, 0f, 0f, -1f, 1f, target))
        // The nearest frontier is at the edge of the carved cone, not inside it or behind the wall.
        assertTrue("target should be in front of the camera", target[2] > 0f)
        assertTrue("target should stop short of the wall", target[2] < 3f)
    }

    @Test
    fun `frontier target is never the free space against the lens`() {
        val grid = VoxelGrid()
        var x = -1f
        while (x <= 1f) {
            grid.integrate(0f, 0f, 0f, x, 0f, 3f)
            x += 0.05f
        }
        val target = FloatArray(3)
        assertTrue(grid.frontierTarget(0f, 0f, 0f, -1f, 1f, target))
        val distance = kotlin.math.sqrt(target[0] * target[0] + target[1] * target[1] + target[2] * target[2])
        assertTrue("target was $distance m away", distance >= VoxelGrid.FRONTIER_MIN_DISTANCE_M)
    }

    @Test
    fun `occlusion target points behind the hit, not into the frontier`() {
        val grid = VoxelGrid()
        // A single wall straight ahead: FREE up to it, OCCUPIED at the hit, OCCLUDED behind.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)

        val target = FloatArray(3)
        assertTrue(grid.occlusionTarget(0f, 0f, 0f, -1f, 1f, target))
        // Must land behind the wall, in the occluded cone -- not in front of it (that's frontier
        // territory) and not past the occlusion depth cap.
        assertTrue("target should be behind the wall", target[2] > 2f)
        assertTrue("target should stay within the occlusion depth", target[2] <= 2f + VoxelGrid.OCCLUSION_DEPTH_M)
    }

    @Test
    fun `no occlusion target without an occluded voxel touching a known surface`() {
        val grid = VoxelGrid()
        // Nothing hit yet: no OCCUPIED voxels exist, so there's nothing for OCCLUDED volume to
        // border, even once free space has been carved.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 40f) // out of MAX_RANGE_M: carves nothing at all
        assertFalse(grid.occlusionTarget(0f, 0f, 0f, -1f, 1f, FloatArray(3)))
    }

    @Test
    fun `no frontier once the free space is closed off`() {
        val grid = VoxelGrid()
        grid.integrate(0f, 0f, 0f, 0f, 0f, 1f)
        // Search band excludes the single carved voxel's height, so nothing qualifies.
        assertFalse(grid.frontierTarget(0f, 0f, 0f, 5f, 6f, FloatArray(3)))
    }

    @Test
    fun `mesh has no interior faces between two adjacent occupied voxels`() {
        val grid = VoxelGrid()
        // Two touching hits, side by side along X: the shared face between them must not appear.
        grid.integrate(0f, 0f, 0f, 0f, 0f, 2f)
        grid.integrate(0.1f, 0f, 0f, 0.1f, 0f, 2f)
        assertEquals(2, grid.occupiedCount)

        val mesh = grid.meshTriangles(10_000)
        assertEquals(0, mesh.size % 40) // whole quads (10 line verts * 4 floats) only
        // Two touching voxels emit 5 exposed quads each (the shared boundary quad is skipped on both sides): 10 quads * 10 verts = 100 verts.
        assertEquals(100, mesh.size / 4)
    }

    @Test
    fun `mesh is empty when nothing is occupied`() {
        val grid = VoxelGrid()
        assertEquals(0, grid.meshTriangles(1000).size)
    }

    @Test
    fun `grid grows past its initial capacity without losing entries`() {
        val grid = VoxelGrid(initialCapacity = 64)
        // 500 distinct surface voxels, all inside the trusted depth range.
        val points = (0 until 500).map {
            Triple((it % 10) * 0.15f - 0.75f, ((it / 10) % 10) * 0.15f, 2f + (it / 100) * 0.2f)
        }
        for ((x, y, z) in points) grid.integrate(0f, 0f, 0f, x, y, z)
        assertEquals(500, grid.occupiedCount)
        assertTrue(grid.capacity > 64)
        assertFalse(grid.full)
        for ((x, y, z) in points) assertEquals(VoxelGrid.OCCUPIED, grid.stateAt(x, y, z))
    }

    private fun sinDeg(d: Float) = kotlin.math.sin(Math.toRadians(d.toDouble())).toFloat()
    private fun cosDeg(d: Float) = kotlin.math.cos(Math.toRadians(d.toDouble())).toFloat()
}
