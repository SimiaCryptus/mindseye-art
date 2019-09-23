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

import com.simiacryptus.mindseye.art.photo.MultivariateFrameOfReference;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.JsonUtil;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public abstract class ContentTopology implements RasterTopology {
  protected final int[] dimensions;
  protected final MultivariateFrameOfReference contentRegion;
  protected final Tensor content;
  private List<double[]> pixels;

  public ContentTopology(Tensor content) {
    this.content = content;
    this.dimensions = content.getDimensions();
    this.contentRegion = new MultivariateFrameOfReference(() -> content.getPixelStream().parallel(), dimensions[2]);
    pixels = IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] coords = getCoordsFromIndex(i);
      return contentRegion.adjust(IntStream.range(0, dimensions[2])
          .mapToDouble(c -> content.get(coords[0], coords[1], c))
          .toArray());
    }).collect(toList());
  }

  public static List<int[]> dual(List<int[]> asymmetric, int[] dimensions) {
    final int[] rows = IntStream.range(0, asymmetric.size()).flatMap(i -> Arrays.stream(asymmetric.get(i)).map(x -> i)).toArray();
    final int[] cols = IntStream.range(0, asymmetric.size()).flatMap(i -> Arrays.stream(asymmetric.get(i))).toArray();
    final Map<Integer, int[]> transposed = IntStream.range(0, cols.length).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> cols[x], Collectors.toList()))
        .entrySet().stream()
        .collect(toMap(x -> x.getKey(), x -> x.getValue().stream().mapToInt(xx -> rows[xx]).toArray()));
    return IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i ->
        IntStream.concat(
            Arrays.stream(asymmetric.get(i)),
            Arrays.stream(transposed.getOrDefault(i, new int[]{}))
        ).distinct().sorted().toArray()).collect(toList());
  }

  @Override
  public abstract List<int[]> connectivity();

  public void log(List<int[]> graph) {
    log(graph, System.out);
  }

  public void log(List<int[]> graph, PrintStream out) {
    out.println("Connectivity Statistics: " + graph.stream().mapToInt(x -> x.length).summaryStatistics());
    out.println("Connectivity Histogram: " + JsonUtil.toJson(graph.stream().collect(Collectors.groupingBy(x -> x.length, Collectors.counting()))));
    out.println("Spatial Distance Statistics: " + IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] pos = getCoordsFromIndex(i);
      return Arrays.stream(graph.get(i)).mapToObj(j -> getCoordsFromIndex(j))
          .mapToDouble(posJ -> IntStream.range(0, pos.length)
              .mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
          .map(Math::sqrt).toArray();
    }).flatMapToDouble(Arrays::stream).summaryStatistics());
    out.println("Spatial Distance Histogram: " + JsonUtil.toJson(IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] pos = getCoordsFromIndex(i);
      return Arrays.stream(graph.get(i)).mapToObj(j -> getCoordsFromIndex(j))
          .mapToDouble(posJ -> IntStream.range(0, pos.length)
              .mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
          .map(Math::sqrt).map(Math::round).mapToInt(x -> (int) x).toArray();
    }).flatMapToInt(Arrays::stream).mapToObj(x -> x).collect(Collectors.groupingBy(x -> x, Collectors.counting()))));
    out.println("Color Distance Statistics: " + IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final double[] pixel = pixel(i);
      return Arrays.stream(graph.get(i)).mapToObj(this::pixel)
          .collect(toList()).stream().mapToDouble(p -> chromaDistance(p, pixel))
          .toArray();
    }).flatMapToDouble(Arrays::stream).summaryStatistics());
  }

  public List<int[]> dual(List<int[]> asymmetric) {
    return dual(asymmetric, dimensions);
  }

  protected double[] pixel(int i) {
    return pixels.get(i);
  }

  protected double chromaDistance(double[] a, double[] b) {
    return contentRegion.dist(IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).toArray());
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
}
