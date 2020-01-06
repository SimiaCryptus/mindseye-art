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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;

import java.io.PrintStream;
import java.util.Arrays;

import static java.util.stream.Collectors.toList;

public abstract @RefAware
class ContentTopology extends ReferenceCountingBase
    implements RasterTopology {
  protected final int[] dimensions;
  protected final MultivariateFrameOfReference contentRegion;
  protected final Tensor content;
  private RefList<double[]> pixels;

  public ContentTopology(Tensor content) {
    this.content = content;
    this.dimensions = content.getDimensions();
    this.contentRegion = new MultivariateFrameOfReference(() -> content.getPixelStream().parallel(), dimensions[2]);
    pixels = RefIntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] coords = getCoordsFromIndex(i);
      return contentRegion.adjust(RefIntStream.range(0, dimensions[2])
          .mapToDouble(c -> content.get(coords[0], coords[1], c)).toArray());
    }).collect(RefCollectors.toList());
  }

  @Override
  public int[] getDimensions() {
    return dimensions;
  }

  public static RefList<int[]> dual(
      RefList<int[]> asymmetric, int[] dimensions) {
    final int[] rows = RefIntStream.range(0, asymmetric.size())
        .flatMap(i -> RefArrays.stream(asymmetric.get(i)).map(x -> i)).toArray();
    final int[] cols = RefIntStream.range(0, asymmetric.size())
        .flatMap(i -> RefArrays.stream(asymmetric.get(i))).toArray();
    final RefMap<Integer, int[]> transposed = RefIntStream
        .range(0, cols.length).mapToObj(x -> x)
        .collect(RefCollectors.groupingBy(x -> cols[x],
            RefCollectors.toList()))
        .entrySet().stream()
        .collect(RefCollectors.toMap(x -> x.getKey(), x -> x.getValue().stream().mapToInt(xx -> rows[xx]).toArray()));
    return RefIntStream.range(0, dimensions[0] * dimensions[1])
        .mapToObj(i -> RefIntStream
            .concat(RefArrays.stream(asymmetric.get(i)),
                RefArrays.stream(transposed.getOrDefault(i, new int[]{})))
            .distinct().sorted().toArray())
        .collect(RefCollectors.toList());
  }

  public static @SuppressWarnings("unused")
  ContentTopology[] addRefs(ContentTopology[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ContentTopology::addRef)
        .toArray((x) -> new ContentTopology[x]);
  }

  public static @SuppressWarnings("unused")
  ContentTopology[][] addRefs(ContentTopology[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ContentTopology::addRefs)
        .toArray((x) -> new ContentTopology[x][]);
  }

  @Override
  public abstract RefList<int[]> connectivity();

  public void log(RefList<int[]> graph) {
    log(graph, com.simiacryptus.ref.wrappers.RefSystem.out);
  }

  public void log(RefList<int[]> graph, PrintStream out) {
    out.println("Connectivity Statistics: " + graph.stream().mapToInt(x -> x.length).summaryStatistics());
    out.println(
        "Connectivity Histogram: " + JsonUtil.toJson(graph.stream().collect(RefCollectors
            .groupingBy(x -> x.length, RefCollectors.counting()))));
    out.println("Spatial Distance Statistics: "
        + RefIntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final int[] pos = getCoordsFromIndex(i);
      return RefArrays.stream(graph.get(i)).mapToObj(j -> getCoordsFromIndex(j))
          .mapToDouble(posJ -> RefIntStream.range(0, pos.length)
              .mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
          .map(Math::sqrt).toArray();
    }).flatMapToDouble(RefArrays::stream).summaryStatistics());
    out.println("Spatial Distance Histogram: " + JsonUtil
        .toJson(RefIntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
          final int[] pos = getCoordsFromIndex(i);
          return RefArrays.stream(graph.get(i)).mapToObj(j -> getCoordsFromIndex(j))
              .mapToDouble(posJ -> RefIntStream.range(0, pos.length)
                  .mapToDouble(c -> pos[c] - posJ[c]).map(x1 -> x1 * x1).sum())
              .map(Math::sqrt).map(Math::round).mapToInt(x -> (int) x).toArray();
        }).flatMapToInt(RefArrays::stream).mapToObj(x -> x)
            .collect(RefCollectors.groupingBy(x -> x,
                RefCollectors.counting()))));
    out.println("Color Distance Statistics: "
        + RefIntStream.range(0, dimensions[0] * dimensions[1]).mapToObj(i -> {
      final double[] pixel = pixel(i);
      return RefArrays.stream(graph.get(i)).mapToObj(this::pixel).collect(toList())
          .stream().mapToDouble(p -> chromaDistance(p, pixel)).toArray();
    }).flatMapToDouble(RefArrays::stream).summaryStatistics());
  }

  public RefList<int[]> dual(RefList<int[]> asymmetric) {
    return dual(asymmetric, dimensions);
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

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ContentTopology addRef() {
    return (ContentTopology) super.addRef();
  }

  protected double[] pixel(int i) {
    return pixels.get(i);
  }

  protected double chromaDistance(double[] a, double[] b) {
    return contentRegion
        .dist(RefIntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).toArray());
  }
}
