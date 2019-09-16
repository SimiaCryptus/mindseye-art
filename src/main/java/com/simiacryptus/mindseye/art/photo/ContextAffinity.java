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
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class ContextAffinity implements RasterAffinity {
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

  private static IntStream expand(List<int[]> graphEdges, IntStream stream, int iterations) {
    if (iterations <= 0) {
      return stream;
    } else if (iterations == 1) {
      return expand(graphEdges, stream);
    } else {
      return expand(graphEdges, expand(graphEdges, stream, iterations - 1));
    }
  }

  private static IntStream expand(List<int[]> graphEdges, IntStream intStream) {
    return intStream.mapToObj(graphEdges::get).flatMapToInt(Arrays::stream).distinct();
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
  protected static SimpleMatrix covariance(SimpleMatrix means, SimpleMatrix rms, Supplier<Stream<double[]>> stream, int size) {
    final SimpleMatrix cov = new SimpleMatrix(size, size);
    IntStream.range(0, size).forEach(c1 ->
        IntStream.range(0, size).forEach(c2 -> {
          final double mean1 = means.get(c1);
          final double mean2 = means.get(c2);
          final double rms1 = rms.get(c1);
          final double rms2 = rms.get(c2);
          final double covariance = stream.get().mapToDouble(p -> ((p[c1] - mean1) / rms1) * ((p[c2] - mean2) / rms2)).average().getAsDouble();
          cov.set(c1, c2, covariance);
        })
    );
    return cov;
  }

  @NotNull
  protected static SimpleMatrix magnitude(SimpleMatrix means, Supplier<Stream<double[]>> stream, int size) {
    final SimpleMatrix rms = new SimpleMatrix(size, 1);
    IntStream.range(0, size).forEach(c -> rms.set(c, 0,
        //        256
//        1.0 * Math.sqrt(neighborhood.stream().mapToDouble(p -> p[c] - means.get(c)).map(p -> p * p).average().getAsDouble())
        Math.sqrt(stream.get().mapToDouble(p -> p[c] - means.get(c)).map(Math::abs).max().getAsDouble())
    ));
    return rms;
  }

  @NotNull
  protected static SimpleMatrix means(Supplier<Stream<double[]>> stream, int size) {
    final SimpleMatrix means = new SimpleMatrix(size, 1);
    IntStream.range(0, size).forEach(c -> means.set(c, 0, stream.get().mapToDouble(p -> p[c]).average().getAsDouble()));
    return means;
  }

  @Override
  public List<double[]> affinityList(List<int[]> graphEdges) {
    final int channels = dimensions[2];
    final int pixels = dimensions[0] * dimensions[1];
    final List<int[]> iteratedGraph = IteratedRasterTopology.iterate(graphEdges, getGraphPower1());
    Region region_global = new Region(() -> content.getPixelStream().parallel(), channels);
    return IntStream.range(0, pixels).parallel().mapToObj(i ->
        Arrays.stream(graphEdges.get(i)).mapToDouble(j -> {
          if (i == j) {
            return 1;
          } else {
            final List<double[]> neighborhood = expand(iteratedGraph, IntStream.of(i, j), getGraphPower2()).mapToObj(this::pixel).collect(Collectors.toList());
            Region mix = new Region(region_global, new Region(() -> neighborhood.stream(), channels), getMixing());
            return dist(
                toMatrix(mix.adjust(pixel(i))),
                toMatrix(mix.adjust(pixel(j))),
                mix.cov,
                neighborhood.size(),
                pixels
            );
          }
        }).toArray()
    ).collect(Collectors.toList());
  }

  protected double[] adjust(double[] pixel_i, SimpleMatrix means, SimpleMatrix rms) {
    return IntStream.range(0, dimensions[2]).mapToDouble(c -> ((pixel_i[c]) - means.get(c)) / rms.get(c)).toArray();
  }

  protected abstract double dist(SimpleMatrix vector_i, SimpleMatrix vector_j, SimpleMatrix cov, int neighborhoodSize, int globalSize);

  protected double[] pixel(int i) {
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
  public ContextAffinity setTopology(RasterTopology topology) {
    this.topology = topology;
    return this;
  }

  public double getMixing() {
    return mixing;
  }

  public ContextAffinity setMixing(double mixing) {
    this.mixing = mixing;
    return this;
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

  public static class Region {
    private SimpleMatrix means;
    private SimpleMatrix rms;
    private SimpleMatrix cov;
    private int dimension;

    public Region(Region a, Region b, double mixing) {
      means = mix(b.means, a.means, mixing);
      rms = mix(b.rms, a.rms, mixing);
      cov = mix(b.cov, a.cov, mixing);
      this.dimension = a.dimension;
    }

    public Region(Supplier<Stream<double[]>> fn2, int channels) {
      this.dimension = channels;
      means = means(fn2, this.dimension);
      rms = magnitude(means, fn2, channels);
      cov = covariance(means, rms, fn2, channels);
    }

    public double[] adjust(double[] pixel) {
      return IntStream.range(0, pixel.length).mapToDouble(c -> ((pixel[c]) - this.means.get(c)) / this.rms.get(c)).toArray();
    }
  }

}
