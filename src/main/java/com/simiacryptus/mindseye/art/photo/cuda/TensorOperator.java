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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public class TensorOperator extends ReferenceCountingBase implements RefOperator<Tensor> {
  private final RefOperator<double[][]> inner;
  private final int[] dimensions;
  private final @RefAware RasterTopology topology;

  public TensorOperator(RefOperator<double[][]> inner, int[] dimensions, @RefAware RasterTopology topology) {
    this.inner = inner;
    this.dimensions = dimensions;
    this.topology = topology;
  }

  @Nullable
  @Override
  public Tensor apply(@Nonnull Tensor tensor) {
    int[] tensorDimensions = tensor.getDimensions();
    if (!RefArrays.equals(this.dimensions, tensorDimensions)) {
      tensor.freeRef();
      throw new IllegalArgumentException(
          RefArrays.toString(this.dimensions) + " != " + RefArrays.toString(tensorDimensions));
    }
    final int channels = 3 <= this.dimensions.length ? this.dimensions[2] : 1;
    final double[][] imageMatrix = RefIntStream.range(0, channels).mapToObj(c -> {
      final double[] doubles = new double[this.dimensions[0] * this.dimensions[1]];
      for (int y = 0; y < this.dimensions[1]; y++) {
        for (int x = 0; x < this.dimensions[0]; x++) {
          doubles[topology.getIndexFromCoords(x, y)] = 3 <= this.dimensions.length ? tensor.get(x, y, c) : tensor.get(x, y);
        }
      }
      return doubles;
    }).toArray(i -> new double[i][]);
    double[][] smoothed = inner.apply(imageMatrix);
    Tensor mapCoords = tensor.mapCoords(coordinate -> {
      final int[] c = coordinate.getCoords();
      final int channel = 3 <= this.dimensions.length ? c[2] : 0;
      return Math.min(Math.max(smoothed[channel][topology.getIndexFromCoords(c[0], c[1])], 0), 255);
    });
    tensor.freeRef();
    return mapCoords;
  }

  public void _free() {
    super._free();
    inner.freeRef();
    RefUtil.freeRef(topology);
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TensorOperator addRef() {
    return (TensorOperator) super.addRef();
  }

}
