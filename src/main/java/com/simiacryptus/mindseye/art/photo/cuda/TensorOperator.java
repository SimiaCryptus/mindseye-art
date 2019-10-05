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

package com.simiacryptus.mindseye.art.photo.cuda;

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

public class TensorOperator extends ReferenceCountingBase implements RefOperator<Tensor> {
  private final RefOperator<double[][]> inner;
  private final int[] dimensions;
  private final RasterTopology topology;

  public TensorOperator(RefOperator<double[][]> inner, int[] dimensions, RasterTopology topology) {
    this.inner = inner;
    this.dimensions = dimensions;
    this.topology = topology;
  }

  @Override
  public Tensor apply(Tensor tensor) {
    if (!Arrays.equals(dimensions, tensor.getDimensions()))
      throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(tensor.getDimensions()));
    final int channels = 3 <= dimensions.length ? dimensions[2] : 1;
    final double[][] imageMatrix = IntStream.range(0, channels).mapToObj(c -> {
      final double[] doubles = new double[dimensions[0] * dimensions[1]];
      for (int y = 0; y < dimensions[1]; y++) {
        for (int x = 0; x < dimensions[0]; x++) {
          doubles[topology.getIndexFromCoords(x, y)] = 3 <= dimensions.length ? tensor.get(x, y, c) : tensor.get(x, y);
        }
      }
      return doubles;
    }).toArray(i -> new double[i][]);
    double[][] smoothed = inner.apply(imageMatrix);
    return tensor.mapCoordsAndFree(coordinate -> {
      final int[] c = coordinate.getCoords();
      final int channel = 3 <= dimensions.length ? c[2] : 0;
      return Math.min(Math.max(smoothed[channel][topology.getIndexFromCoords(c[0], c[1])], 0), 255);
    });
  }

  @Override
  protected void _free() {
    inner.freeRef();
    super._free();
  }

}
