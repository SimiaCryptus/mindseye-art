/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.photo.topology;

import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

public class SimpleRasterTopology implements RasterTopology {
  protected final int[] dimensions;
  private final int max_neighborhood_size = 9;

  public SimpleRasterTopology(int[] dimensions) {
    this.dimensions = dimensions;
  }

  @Override
  public int[] getDimensions() {
    return dimensions;
  }

  @Override
  public RefList<int[]> connectivity() {
    final ThreadLocal<int[]> neighbors = ThreadLocal.withInitial(() -> new int[max_neighborhood_size]);
    return RefIntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i -> {
      final int[] original = neighbors.get();
      return RefArrays.copyOf(original, getNeighbors(getCoordsFromIndex(i), original));
    }).collect(RefCollectors.toList());
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return x + dimensions[0] * y;
  }

  @Override
  public int[] getCoordsFromIndex(int i) {
    final int x = i % dimensions[0];
    final int y = (i - x) / dimensions[0];
    return new int[] { x, y };
  }

  private int getNeighbors(int[] coords, int[] neighbors) {
    int neighborCount = 0;
    final int x = coords[0];
    final int y = coords[1];
    final int w = dimensions[0];
    final int h = dimensions[1];
    if (y > 0) {
      if (x > 0) {
        neighbors[neighborCount++] = getIndexFromCoords(x - 1, y - 1);
      }
      if (x < w - 1) {
        neighbors[neighborCount++] = getIndexFromCoords(x + 1, y - 1);
      }
      neighbors[neighborCount++] = getIndexFromCoords(x, y - 1);
    }
    if (y < h - 1) {
      if (x > 0) {
        neighbors[neighborCount++] = getIndexFromCoords(x - 1, y + 1);
      }
      if (x < w - 1) {
        neighbors[neighborCount++] = getIndexFromCoords(x + 1, y + 1);
      }
      neighbors[neighborCount++] = getIndexFromCoords(x, y + 1);
    }
    if (x > 0) {
      neighbors[neighborCount++] = getIndexFromCoords(x - 1, y);
    }
    if (x < w - 1) {
      neighbors[neighborCount++] = getIndexFromCoords(x + 1, y);
    }
    //  neighbors[neighborCount++] = getIndexFromCoords(x, y);
    return neighborCount;
  }
}
