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

import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.ReferenceCountingBase;

public @com.simiacryptus.ref.lang.RefAware
class TensorOperator extends ReferenceCountingBase
    implements RefOperator<Tensor> {
  private final RefOperator<double[][]> inner;
  private final int[] dimensions;
  private final RasterTopology topology;

  public TensorOperator(RefOperator<double[][]> inner, int[] dimensions, RasterTopology topology) {
    this.inner = inner;
    this.dimensions = dimensions;
    this.topology = topology;
  }

  public static @SuppressWarnings("unused")
  TensorOperator[] addRefs(TensorOperator[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(TensorOperator::addRef)
        .toArray((x) -> new TensorOperator[x]);
  }

  public static @SuppressWarnings("unused")
  TensorOperator[][] addRefs(TensorOperator[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(TensorOperator::addRefs)
        .toArray((x) -> new TensorOperator[x][]);
  }

  @Override
  public Tensor apply(Tensor tensor) {
    if (!com.simiacryptus.ref.wrappers.RefArrays.equals(dimensions, tensor.getDimensions()))
      throw new IllegalArgumentException(com.simiacryptus.ref.wrappers.RefArrays.toString(dimensions) + " != "
          + com.simiacryptus.ref.wrappers.RefArrays.toString(tensor.getDimensions()));
    final int channels = 3 <= dimensions.length ? dimensions[2] : 1;
    final double[][] imageMatrix = com.simiacryptus.ref.wrappers.RefIntStream.range(0, channels).mapToObj(c -> {
      final double[] doubles = new double[dimensions[0] * dimensions[1]];
      for (int y = 0; y < dimensions[1]; y++) {
        for (int x = 0; x < dimensions[0]; x++) {
          doubles[topology.getIndexFromCoords(x, y)] = 3 <= dimensions.length ? tensor.get(x, y, c) : tensor.get(x, y);
        }
      }
      return doubles;
    }).toArray(i -> new double[i][]);
    double[][] smoothed = inner.apply(imageMatrix);
    return tensor.mapCoords(coordinate -> {
      final int[] c = coordinate.getCoords();
      final int channel = 3 <= dimensions.length ? c[2] : 0;
      return Math.min(Math.max(smoothed[channel][topology.getIndexFromCoords(c[0], c[1])], 0), 255);
    });
  }

  public void _free() {
    inner.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  TensorOperator addRef() {
    return (TensorOperator) super.addRef();
  }

}
