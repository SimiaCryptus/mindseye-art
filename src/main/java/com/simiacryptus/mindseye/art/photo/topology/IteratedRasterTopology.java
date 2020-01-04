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

public @com.simiacryptus.ref.lang.RefAware
class IteratedRasterTopology implements RasterTopology {
  private final RasterTopology inner;
  private int iterations = 2;

  public IteratedRasterTopology(RasterTopology inner) {
    this.inner = inner;
    this.setIterations(getIterations());
  }

  @Override
  public int[] getDimensions() {
    return inner.getDimensions();
  }

  public int getIterations() {
    return iterations;
  }

  public IteratedRasterTopology setIterations(int iterations) {
    this.iterations = iterations;
    return this;
  }

  public static com.simiacryptus.ref.wrappers.RefList<int[]> iterate(com.simiacryptus.ref.wrappers.RefList<int[]> edges,
                                                                     int pow) {
    assert pow > 0;
    if (1 == pow) {
      return edges;
    } else {
      final com.simiacryptus.ref.wrappers.RefList<int[]> prev = iterate(edges, pow - 1);
      return com.simiacryptus.ref.wrappers.RefIntStream.range(0, prev.size()).parallel()
          .mapToObj(j -> com.simiacryptus.ref.wrappers.RefArrays.stream(prev.get(j))
              .flatMap(i -> com.simiacryptus.ref.wrappers.RefArrays.stream(prev.get(i))).filter(i -> i != j).distinct()
              .toArray())
          .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    }
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<int[]> connectivity() {
    return iterate(inner.connectivity(), getIterations());
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return inner.getIndexFromCoords(x, y);
  }

  @Override
  public int[] getCoordsFromIndex(int i) {
    return inner.getCoordsFromIndex(i);
  }
}
