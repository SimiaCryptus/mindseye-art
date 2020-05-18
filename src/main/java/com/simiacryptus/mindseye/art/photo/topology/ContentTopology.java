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
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

/**
 * The type Content topology.
 */
public abstract class ContentTopology extends ReferenceCountingBase implements RasterTopology {
  /**
   * The Dimensions.
   */
  @Nonnull
  protected final int[] dimensions;
  /**
   * The Content region.
   */
  @Nonnull
  protected final MultivariateFrameOfReference contentRegion;
  /**
   * The Content.
   */
  @Nonnull
  protected final Tensor content;
  private List<double[]> pixels;

  /**
   * Instantiates a new Content topology.
   *
   * @param content the content
   */
  public ContentTopology(@Nonnull Tensor content) {
    this.content = content;
    this.dimensions = this.content.getDimensions();
    this.contentRegion = new MultivariateFrameOfReference(() -> {
      Stream<double[]> stream = ContentTopology.this.content.getPixelStream();
      if (!CoreSettings.INSTANCE().singleThreaded) stream = stream.parallel();
      return stream;
    }, dimensions[2]);
    pixels = IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] coords = getCoordsFromIndex(i);
      return contentRegion.adjust(
          IntStream.range(0, dimensions[2]).mapToDouble(c -> ContentTopology.this.content.get(coords[0], coords[1], c)).toArray());
    }).collect(Collectors.toList());
  }

  @Nonnull
  @Override
  public int[] getDimensions() {
    return dimensions;
  }

  /**
   * Dual list.
   *
   * @param asymmetric the asymmetric
   * @param dimensions the dimensions
   * @return the list
   */
  public static List<int[]> dual(@Nonnull List<int[]> asymmetric, int[] dimensions) {
    final int[] rows = IntStream.range(0, asymmetric.size()).flatMap(i -> {
      int[] data2 = asymmetric.get(i);
      return Arrays.stream(data2).map(x -> i);
    }).toArray();
    final int[] cols = IntStream.range(0, asymmetric.size()).flatMap(i -> {
      int[] data2 = asymmetric.get(i);
      return Arrays.stream(data2);
    }).toArray();
    Map<Integer, List<Integer>> map = IntStream.range(0, cols.length).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> cols[x], Collectors.toList()));
    Set<Map.Entry<Integer, List<Integer>>> entries = map.entrySet();
    final Map<Integer, int[]> transposed = entries.stream()
        .collect(Collectors.toMap(entry -> {
          Integer key = entry.getKey();
          RefUtil.freeRef(entry);
          return key;
        }, entry -> {
          int[] ints = entry.getValue().stream().mapToInt(xx -> rows[xx]).toArray();
          RefUtil.freeRef(entry);
          return ints;
        }));
    List<int[]> list = IntStream.range(0, dimensions[0] * dimensions[1])
        .mapToObj(i -> {
          int[] data = transposed.getOrDefault(i, new int[]{});
          int[] data1 = asymmetric.get(i);
          return IntStream
              .concat(Arrays.stream(data1), Arrays.stream(data))
              .distinct().sorted().toArray();
        }).collect(toList());
    return list;
  }

  @Override
  public abstract List<int[]> connectivity();

  /**
   * Log.
   *
   * @param graph the graph
   */
  public void log(@Nonnull List<int[]> graph) {
    log(graph, System.out);
  }

  /**
   * Log.
   *
   * @param graph the graph
   * @param out   the out
   */
  public void log(@Nonnull List<int[]> graph, @Nonnull PrintStream out) {
    out.println("Connectivity Statistics: " + graph.stream().mapToInt(x -> x.length).summaryStatistics());
    out.println("Connectivity Histogram: "
        + JsonUtil.toJson(graph.stream().collect(Collectors.groupingBy(x -> x.length, Collectors.counting()))));
    out.println("Spatial Distance Statistics: " + IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] pos = getCoordsFromIndex(i);
      int[] data = graph.get(i);
      return Arrays.stream(data).mapToObj(j -> getCoordsFromIndex(j))
          .mapToDouble(
              posJ -> IntStream.range(0, pos.length).mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
          .map(a -> Math.sqrt(a)).toArray();
    }).flatMapToDouble(data2 -> Arrays.stream(data2)).summaryStatistics());
    out.println("Spatial Distance Histogram: "
        + JsonUtil.toJson(IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] pos = getCoordsFromIndex(i);
      int[] data = graph.get(i);
      return Arrays.stream(data).mapToObj(j -> getCoordsFromIndex(j))
          .mapToDouble(
              posJ -> IntStream.range(0, pos.length).mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
          .map(a1 -> Math.sqrt(a1)).map(a -> Math.round(a)).mapToInt(x -> (int) x).toArray();
    }).flatMapToInt(data1 -> Arrays.stream(data1)).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> x, Collectors.counting()))));
    out.println("Color Distance Statistics: " + IntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final double[] pixel = pixel(i);
      int[] data = graph.get(i);
      return Arrays.stream(data).mapToObj(i1 -> pixel(i1)).collect(toList()).stream()
          .mapToDouble(p -> chromaDistance(p, pixel)).toArray();
    }).flatMapToDouble(data -> Arrays.stream(data)).summaryStatistics());
  }

  /**
   * Dual list.
   *
   * @param asymmetric the asymmetric
   * @return the list
   */
  public List<int[]> dual(@Nonnull List<int[]> asymmetric) {
    return dual(asymmetric, dimensions);
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return x + dimensions[0] * y;
  }

  @Nonnull
  @Override
  public int[] getCoordsFromIndex(int i) {
    final int x = i % dimensions[0];
    final int y = (i - x) / dimensions[0];
    return new int[]{x, y};
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    content.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ContentTopology addRef() {
    return (ContentTopology) super.addRef();
  }

  /**
   * Pixel double [ ].
   *
   * @param i the
   * @return the double [ ]
   */
  protected double[] pixel(int i) {
    return pixels.get(i);
  }

  /**
   * Chroma distance double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  protected double chromaDistance(@Nonnull double[] a, double[] b) {
    return contentRegion.dist(IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).toArray());
  }
}
