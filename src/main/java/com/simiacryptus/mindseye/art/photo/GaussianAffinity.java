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

import com.simiacryptus.mindseye.lang.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implements Matting Affinity
 * <p>
 * See Also: A Closed Form Solution to Natural Image Matting
 * http://cs.brown.edu/courses/cs129/results/final/valayshah/Matting-Levin-Lischinski-Weiss-CVPR06.pdf
 * <p>
 * Replaced with experimental metric
 * See Also: https://en.wikipedia.org/wiki/Mahalanobis_distance
 */
public class GaussianAffinity implements RasterAffinity {
  protected final Tensor content;
  private final double sigma;
  private final int[] dimensions;
  private RasterTopology topology;

  public GaussianAffinity(Tensor content, double sigma) {
    this(content, sigma, new SimpleRasterTopology(content.getDimensions()));
  }

  public GaussianAffinity(Tensor content, double sigma, RasterTopology topology) {
    this.setTopology(topology);
    this.dimensions = content.getDimensions();
    this.content = content;
    this.sigma = sigma;
  }

  @Override
  public List<double[]> affinityList(List<int[]> graphEdges) {
    return IntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i ->
        Arrays.stream(graphEdges.get(i)).mapToDouble(j ->
            affinity(i, j)).toArray()
    ).collect(Collectors.toList());
  }

  protected double affinity(int i, int j) {
    final double[] pixel_i = pixel(i);
    final double[] pixel_j = pixel(j);
    return Math.exp(-IntStream.range(0, pixel_i.length).mapToDouble(idx -> pixel_i[idx] - pixel_j[idx]).map(x -> x * x).sum() / (sigma * sigma));
  }

  private double[] pixel(int i) {
    final int[] coords = getTopology().getCoordsFromIndex(i);
    return IntStream.range(0, dimensions[2]).mapToDouble(c -> {
      return content.get(coords[0], coords[1], c);
    }).toArray();
  }

  @Override
  public RasterTopology getTopology() {
    return topology;
  }

  @Override
  public GaussianAffinity setTopology(RasterTopology topology) {
    this.topology = topology;
    return this;
  }
}

