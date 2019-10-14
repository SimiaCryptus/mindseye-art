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

import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.RefOperator;
import com.simiacryptus.mindseye.art.photo.cuda.SparseMatrixFloat;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.util.FastRandom;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.adjust;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.degree;

public class SegmentUtil {

  @NotNull
  public static Tensor resize(Tensor tensor, int imageSize) {
    final Tensor resized = Tensor.fromRGB(
        ImageUtil.resize(
            tensor.toImage(),
            imageSize,
            true)
    );
    tensor.freeRef();
    return resized;
  }

  public static int[] valueCountArray(int[] ints) {
    final Map<Integer, Long> countMap = valueCountMap(ints);
    return IntStream.range(0, Arrays.stream(ints).max().getAsInt() + 1).map((int i) -> (int) (long) countMap.getOrDefault(i, 0l)).toArray();
  }

  public static Map<Integer, Long> valueCountMap(int[] ints) {
    return Arrays.stream(ints).mapToObj(x -> x).collect(Collectors.groupingBy((Integer x) -> x, Collectors.counting()));
  }

  public static void printHistogram(int[] islands) {
    Arrays.stream(islands).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> x, Collectors.counting()))
        .values().stream()
        .collect(Collectors.groupingBy(x -> x, Collectors.counting()))
        .entrySet().stream()
        .sorted(Comparator.comparing(x -> -x.getValue() * x.getKey()))
        .map(x -> String.format("%d regions of size %s", x.getValue(), x.getKey()))
        .forEach(System.out::println);
  }

  public static <T> int[] markIslands(RasterTopology topology, Function<int[], T> extract, BiPredicate<T, T> test, int maxRecursion, int rows) {
    int[] marks = new int[rows];
    int islandNumber = 0;
    int[] dimensions = topology.getDimensions();
    for (int x = 0; x < dimensions[0]; x++) {
      for (int y = 0; y < dimensions[1]; y++) {
        islandNumber = _markIslands(topology, extract, test, marks, islandNumber, maxRecursion, x, y);
      }
    }
    return marks;
  }

  protected static <T> int _markIslands(RasterTopology topology, Function<int[], T> extract, BiPredicate<T, T> test, int[] marks, int islandNumber, int maxRecursion, int... coords) {
    final int row = topology.getIndexFromCoords(coords[0], coords[1]);
    final int thisIsland;
    if (marks[row] == 0) {
      thisIsland = ++islandNumber;
      marks[row] = thisIsland;
    } else {
      thisIsland = marks[row];
    }
    assert 0 < thisIsland;
    final T rowColor = extract.apply(coords);
    final List<int[]> connectivity = topology.connectivity();
    if (maxRecursion > 0) for (int col : connectivity.get(row)) {
      if (0 == marks[col]) {
        final int[] toCoords = topology.getCoordsFromIndex(col);
        if (test.test(rowColor, extract.apply(toCoords))) {
          marks[col] = thisIsland;
          islandNumber = _markIslands(topology, extract, test, marks, islandNumber, maxRecursion - 1, toCoords);
        }
      }
    }
    return islandNumber;
  }

  public static int[] removeTinyInclusions(int[] pixelMap, SparseMatrixFloat graph, int sizeThreshold) {
    return removeTinyInclusions(pixelMap, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(int[] pixelMap, SparseMatrixFloat graph, int smallSize, int largeSize) {
    return removeTinyInclusions(valueCountMap(pixelMap), graph, smallSize, largeSize);
  }

  public static int[] removeTinyInclusions(Map<Integer, Long> islandSizes, SparseMatrixFloat graph, int sizeThreshold) {
    return removeTinyInclusions(islandSizes, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(Map<Integer, Long> islandSizes, SparseMatrixFloat graph, int smallSize, int largeSize) {
    return IntStream.range(0, graph.rows).map(row -> {
      final int[] cols = graph.getCols(row);
      if (islandSizes.getOrDefault(row, 0l) < smallSize) {
        final int[] largeNeighbors = Arrays.stream(cols).filter(j -> {
          return islandSizes.getOrDefault(j, 0l) >= largeSize;
        }).toArray();
        if (largeNeighbors.length == 1) {
          return largeNeighbors[0];
        }
      }
      return row;
    }).toArray();
  }

  @NotNull
  public static BufferedImage flattenColors(Tensor content, RasterTopology topology, RasterAffinity affinity, int n, SmoothSolver solver) {
    final RefOperator<Tensor> refOperator = solver.solve(
        topology, affinity.wrap((graphEdges, innerResult) -> adjust(
            graphEdges,
            innerResult,
            degree(innerResult),
            0.5)),
        1e-4);
    final Tensor tensor = refOperator.iterate(n, content.addRef());
    refOperator.freeRef();
    final BufferedImage image = tensor.toRgbImage();
    tensor.freeRef();
    return image;
  }

  public static BufferedImage paintWithRandomColors(RasterTopology topology, Tensor content, int[] pixelMap, SparseMatrixFloat graph) {
    return paint(topology, content, pixelMap, randomColors(graph, 0));
  }

  private static Map<Integer, double[]> randomColors(SparseMatrixFloat graph, int iterations) {
    return randomColors(
        graph,
        x -> DoubleStream.generate(() -> FastRandom.INSTANCE.random() * 255).limit(3).toArray(),
        new ConcurrentHashMap<>(),
        iterations
    );
  }

  private static Map<Integer, double[]> randomColors(SparseMatrixFloat graph, Function<Integer, double[]> seedColor, Map<Integer, double[]> colors, int n) {
    if (n <= 0) {
      if (colors.isEmpty()) {
        return Arrays.stream(graph.activeRows()).mapToObj(x -> x).collect(Collectors.toMap(x -> x, seedColor));
      } else {
        return colors;
      }
    }
    return randomColors(graph, seedColor, iterateColors(graph, seedColor, colors), n - 1);
  }

  private static Map<Integer, double[]> iterateColors(SparseMatrixFloat graph, Function<Integer, double[]> seedColor, Map<Integer, double[]> colors) {
    return Arrays.stream(graph.activeRows()).parallel().mapToObj(x -> x).collect(Collectors.toMap(
        key -> key, key -> {
          final int[] cols = graph.getCols(key);
          final float[] vals = graph.getVals(key);
          final List<double[]> neighborColors = IntStream.range(0, cols.length)
              .mapToObj(ni -> Arrays.stream(colors.computeIfAbsent(cols[ni], seedColor))
                  .map(x -> x / vals[ni]).toArray())
              .collect(Collectors.toList());
          if (neighborColors.isEmpty()) return colors.computeIfAbsent(key, seedColor);
          final double[] average = IntStream.range(0, 3).mapToDouble(i -> -neighborColors.stream().mapToDouble(j -> j[i] - 127).average().orElse(0.0)).toArray();
          final double rms = Math.sqrt(Arrays.stream(average).map(x -> x * x).average().getAsDouble());
          return Arrays.stream(average).map(x -> Math.min(Math.max((x / rms) * 64 + 127, 0), 255)).toArray();
        }
    ));
  }

  @NotNull
  public static BufferedImage paint(RasterTopology topology, Tensor content, int[] pixelMap, Map<Integer, double[]> colors) {
    final Tensor tensor = content.mapCoords(c -> {
      final int[] coords = c.getCoords();
      final int regionId = pixelMap[topology.getIndexFromCoords(coords[0], coords[1])];
      final double[] color = colors.get(regionId);
      return null == color ? 0 : color[coords[2]];
    });
    final BufferedImage image = tensor.toImage();
    tensor.freeRef();
    return image;
  }

}
