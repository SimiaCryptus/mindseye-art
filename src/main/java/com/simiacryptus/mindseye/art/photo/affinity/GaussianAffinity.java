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

package com.simiacryptus.mindseye.art.photo.affinity;

import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.SimpleRasterTopology;
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GaussianAffinity extends ReferenceCountingBase implements RasterAffinity {
  @Nonnull
  protected final Tensor content;
  private final double sigma;
  @Nonnull
  private final int[] dimensions;
  private RasterTopology topology;

  public GaussianAffinity(@Nonnull Tensor content, double sigma) {
    this(content, sigma, new SimpleRasterTopology(content.getDimensions()));
  }

  public GaussianAffinity(@Nonnull Tensor content, double sigma, RasterTopology topology) {
    setTopology(topology);
    this.dimensions = content.getDimensions();
    this.content = content;
    this.sigma = sigma;
  }

  public RasterTopology getTopology() {
    return topology.addRef();
  }

  public void setTopology(RasterTopology topology) {
    this.topology = topology;
  }

  @Override
  public List<double[]> affinityList(@Nonnull List<int[]> graphEdges) {
    IntStream stream = IntStream.range(0, dimensions[0] * dimensions[1]);
    if (!CoreSettings.INSTANCE().isSingleThreaded()) stream = stream.parallel();
    return stream
        .mapToObj(i -> Arrays.stream(graphEdges.get(i)).mapToDouble(j -> affinity(i, j)).toArray())
        .collect(Collectors.toList());
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    content.freeRef();
    topology.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GaussianAffinity addRef() {
    return (GaussianAffinity) super.addRef();
  }

  protected double affinity(int i, int j) {
    final double[] pixel_i = pixel(i);
    final double[] pixel_j = pixel(j);
    return Math.exp(
        -IntStream.range(0, pixel_i.length).mapToDouble(idx -> pixel_i[idx] - pixel_j[idx]).map(x -> x * x).sum()
            / (sigma * sigma));
  }

  private double[] pixel(int i) {
    RasterTopology topology = getTopology();
    final int[] coords = topology.getCoordsFromIndex(i);
    topology.freeRef();
    return IntStream.range(0, dimensions[2]).mapToDouble(c -> {
      return content.get(coords[0], coords[1], c);
    }).toArray();
  }
}
