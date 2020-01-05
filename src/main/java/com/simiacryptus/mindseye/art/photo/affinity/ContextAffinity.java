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

import com.simiacryptus.mindseye.art.photo.MultivariateFrameOfReference;
import com.simiacryptus.mindseye.art.photo.topology.IteratedRasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.function.Supplier;

public abstract @RefAware
class ContextAffinity extends ReferenceCountingBase
    implements RasterAffinity {
  protected final Tensor content;
  protected final int[] dimensions;
  private RasterTopology topology;
  private double mixing = 0.5;
  private int graphPower1 = 3;
  private int graphPower2 = 2;

  public ContextAffinity(Tensor content) {
    this.content = content;
    this.dimensions = content.getDimensions();
  }

  public int getGraphPower1() {
    return graphPower1;
  }

  public ContextAffinity setGraphPower1(int graphPower1) {
    this.graphPower1 = graphPower1;
    return this;
  }

  public int getGraphPower2() {
    return graphPower2;
  }

  public ContextAffinity setGraphPower2(int graphPower2) {
    this.graphPower2 = graphPower2;
    return this;
  }

  public double getMixing() {
    return mixing;
  }

  public ContextAffinity setMixing(double mixing) {
    this.mixing = mixing;
    return this;
  }

  public RasterTopology getTopology() {
    return topology;
  }

  public ContextAffinity setTopology(RasterTopology topology) {
    this.topology = topology;
    return this;
  }

  public static SimpleMatrix toMatrix(double[] array) {
    return new SimpleMatrix(array.length, 1, false, array);
  }

  public static SimpleMatrix mix(SimpleMatrix means_local, SimpleMatrix means_global, double pos) {
    assert pos >= 0;
    assert pos <= 1;
    return means_local.scale(pos).plus(means_global.scale(1 - pos));
  }

  @NotNull
  public static SimpleMatrix covariance(SimpleMatrix means, SimpleMatrix rms,
                                        Supplier<RefStream<double[]>> stream, int size) {
    final SimpleMatrix cov = new SimpleMatrix(size, size);
    RefIntStream.range(0, size).parallel().forEach(c1 -> {
      final double mean1 = means.get(c1);
      final double rms1 = rms.get(c1);
      RefIntStream.range(0, size).forEach(c2 -> {
        final double mean2 = means.get(c2);
        final double rms2 = rms.get(c2);
        final double covariance = stream.get().mapToDouble(p -> ((p[c1] - mean1) / rms1) * ((p[c2] - mean2) / rms2))
            .average().getAsDouble();
        cov.set(c1, c2, covariance);
      });
    });
    return cov;
  }

  @NotNull
  public static SimpleMatrix magnitude(SimpleMatrix means,
                                       Supplier<RefStream<double[]>> stream, int size) {
    final SimpleMatrix rms = new SimpleMatrix(size, 1);
    RefIntStream.range(0, size).forEach(c -> rms.set(c, 0,
        //        256
        //        1.0 * Math.sqrt(neighborhood.stream().mapToDouble(p -> p[c] - means.get(c)).map(p -> p * p).average().getAsDouble())
        Math.sqrt(stream.get().mapToDouble(p -> p[c] - means.get(c)).map(Math::abs).max().getAsDouble())));
    return rms;
  }

  @NotNull
  public static SimpleMatrix means(Supplier<RefStream<double[]>> stream, int size) {
    final SimpleMatrix means = new SimpleMatrix(size, 1);
    RefIntStream.range(0, size)
        .forEach(c -> means.set(c, 0, stream.get().mapToDouble(p -> p[c]).average().orElse(0)));
    return means;
  }

  public static @SuppressWarnings("unused")
  ContextAffinity[] addRefs(ContextAffinity[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ContextAffinity::addRef)
        .toArray((x) -> new ContextAffinity[x]);
  }

  public static @SuppressWarnings("unused")
  ContextAffinity[][] addRefs(ContextAffinity[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ContextAffinity::addRefs)
        .toArray((x) -> new ContextAffinity[x][]);
  }

  private static RefIntStream expand(
      RefList<int[]> graphEdges, RefIntStream stream,
      int iterations) {
    if (iterations <= 0) {
      return stream;
    } else if (iterations == 1) {
      return expand(graphEdges, stream);
    } else {
      return expand(graphEdges, expand(graphEdges, stream, iterations - 1));
    }
  }

  private static RefIntStream expand(
      RefList<int[]> graphEdges, RefIntStream intStream) {
    return intStream.mapToObj(graphEdges::get).flatMapToInt(RefArrays::stream).distinct();
  }

  @Override
  public RefList<double[]> affinityList(
      RefList<int[]> graphEdges) {
    final int channels = dimensions[2];
    final int pixels = dimensions[0] * dimensions[1];
    final RefList<int[]> iteratedGraph = IteratedRasterTopology.iterate(graphEdges,
        getGraphPower1());
    MultivariateFrameOfReference region_global = new MultivariateFrameOfReference(
        () -> content.getPixelStream().parallel(), channels);
    return RefIntStream.range(0, pixels).parallel()
        .mapToObj(i -> RefArrays.stream(graphEdges.get(i)).mapToDouble(j -> {
          if (i == j) {
            return 1;
          } else {
            final RefList<double[]> neighborhood = expand(iteratedGraph,
                RefIntStream.of(i, j), getGraphPower2()).mapToObj(this::pixel)
                .collect(RefCollectors.toList());
            if (neighborhood.isEmpty())
              return 1;
            MultivariateFrameOfReference mix = new MultivariateFrameOfReference(region_global,
                new MultivariateFrameOfReference(() -> neighborhood.stream(), channels), getMixing());
            return dist(toMatrix(mix.adjust(pixel(i))), toMatrix(mix.adjust(pixel(j))), mix.cov, neighborhood.size(),
                pixels);
          }
        }).toArray()).collect(RefCollectors.toList());
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ContextAffinity addRef() {
    return (ContextAffinity) super.addRef();
  }

  protected double[] adjust(double[] pixel_i, SimpleMatrix means, SimpleMatrix rms) {
    return RefIntStream.range(0, dimensions[2])
        .mapToDouble(c -> ((pixel_i[c]) - means.get(c)) / rms.get(c)).toArray();
  }

  protected abstract double dist(SimpleMatrix vector_i, SimpleMatrix vector_j, SimpleMatrix cov, int neighborhoodSize,
                                 int globalSize);

  protected double[] pixel(int i) {
    final int[] coords = getTopology().getCoordsFromIndex(i);
    return RefIntStream.range(0, dimensions[2]).mapToDouble(c -> {
      return content.get(coords[0], coords[1], c);
    }).toArray();
  }

}
