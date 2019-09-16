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

package com.simiacryptus.mindseye.art.photo;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
  public List<int[]> connectivity() {
    final int maxRadius = (int) Math.ceil(this.maxRadius);
    final double maxSq = this.maxRadius * this.maxRadius;
    final double minSq = Math.signum(this.minRadius) * (this.minRadius * this.minRadius);
    return IntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i -> {
      final int[] coordsFromIndex = getCoordsFromIndex(i);
      final int[] ints = IntStream.range(-maxRadius, maxRadius).flatMap(x -> {
        final int xx = x + coordsFromIndex[0];
        return IntStream.range(-maxRadius, maxRadius)
            .filter(y -> {
              final int radiusSq = x * x + y * y;
              return radiusSq > minSq && radiusSq <= maxSq;
            })
            .map(y -> y + coordsFromIndex[1])
            .filter(yy -> yy >= 0 && xx >= 0)
            .filter(yy -> yy < dimensions[1] && xx < dimensions[0])
            .map(yy -> getIndexFromCoords(xx, yy));
      }).toArray();
      return ints;
    }).collect(Collectors.toList());
  }

  @Override
  public int[] getDimensions() {
    return dimensions;
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return x + dimensions[0] * y;
  }

  @Override
  public int[] getCoordsFromIndex(int i) {
    final int x = i % dimensions[0];
    final int y = (i - x) / dimensions[0];
    return new int[]{x, y};
  }
//
//  @Override
//  public int getIndexFromCoords(int x, int y) {
//    return y + dimensions[1] * x;
//  }
//
//  @Override
//  public int[] getCoordsFromIndex(int i) {
//    final int y = i % dimensions[1];
//    final int x = (i - y) / dimensions[1];
//    return new int[]{x, y};
//  }
}
