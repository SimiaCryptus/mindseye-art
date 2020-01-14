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

import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;

public class RadiusRasterTopology implements RasterTopology {
  protected final int[] dimensions;
  private final double maxRadius;
  private final double minRadius;

  public RadiusRasterTopology(int[] dimensions, double maxRadius, double minRadius) {
    this.dimensions = dimensions;
    this.maxRadius = maxRadius;
    this.minRadius = minRadius;
  }

  @Override
  public int[] getDimensions() {
    return dimensions;
  }

  public static double getRadius(int distA, int distB) {
    return Math.sqrt(distA * distA + distB * distB) + 1e-4;
  }

  @Override
  public RefList<int[]> connectivity() {
    final int maxRadius = (int) Math.ceil(this.maxRadius);
    final double maxSq = this.maxRadius * this.maxRadius;
    final double minSq = Math.signum(this.minRadius) * (this.minRadius * this.minRadius);
    return RefIntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i -> {
      final int[] coordsFromIndex = getCoordsFromIndex(i);
      final int[] ints = RefIntStream.range(-maxRadius, maxRadius).flatMap(x -> {
        final int xx = x + coordsFromIndex[0];
        return RefIntStream.range(-maxRadius, maxRadius).filter(y -> {
          final int radiusSq = x * x + y * y;
          return radiusSq > minSq && radiusSq <= maxSq;
        }).map(y -> y + coordsFromIndex[1]).filter(yy -> yy >= 0 && xx >= 0)
            .filter(yy -> yy < dimensions[1] && xx < dimensions[0]).map(yy -> getIndexFromCoords(xx, yy));
      }).toArray();
      return ints;
    }).collect(RefCollectors.toList());
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return x + dimensions[0] * y;
  }

  @Nonnull
  @Override
  public int[] getCoordsFromIndex(int i) {
    final int x = i % dimensions[0];
    final int y = (i - x) / dimensions[0];
    return new int[]{x, y};
  }
}
