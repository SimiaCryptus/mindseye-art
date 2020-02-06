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
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.FastRandom;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiPredicate;
import java.util.function.Function;

import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.adjust;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.degree;
import static java.util.stream.IntStream.range;

public class SegmentUtil {

  @Nonnull
  public static Tensor resize(@Nonnull Tensor tensor, int imageSize) {
    final Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), imageSize, true));
    tensor.freeRef();
    return resized;
  }

  public static int[] valueCountArray(@Nonnull int[] ints) {
    final RefMap<Integer, Long> countMap = valueCountMap(ints);
    return range(0, RefArrays.stream(ints).max().getAsInt() + 1)
        .map((int i) -> (int) (long) countMap.getOrDefault(i, 0l)).toArray();
  }

  public static RefMap<Integer, Long> valueCountMap(@Nonnull int[] ints) {
    return RefArrays.stream(ints).mapToObj(x -> x)
        .collect(RefCollectors.groupingBy((Integer x) -> x, RefCollectors.counting()));
  }

  public static void printHistogram(@Nonnull int[] islands) {
    RefArrays.stream(islands).mapToObj(x -> x).collect(RefCollectors.groupingBy(x -> x, RefCollectors.counting()))
        .values().stream().collect(RefCollectors.groupingBy(x -> x, RefCollectors.counting())).entrySet().stream()
        .sorted(RefComparator.comparingDouble(x -> -x.getValue() * x.getKey()))
        .map(x -> RefString.format("%d regions of size %s", x.getValue(), x.getKey()))
        .forEach(x1 -> RefSystem.out.println(x1));
  }

  @Nonnull
  public static <T> int[] markIslands(@Nonnull RasterTopology topology, @Nonnull Function<int[], T> extract, @Nonnull BiPredicate<T, T> test,
                                      int maxRecursion, int rows) {
    int[] marks = new int[rows];
    AtomicInteger islandNumber = new AtomicInteger(0);
    int[] dimensions = topology.getDimensions();
    range(0, dimensions[0]).parallel().mapToObj(x -> x).sorted(RefComparator.comparingInt(x -> x.hashCode()))
        .mapToInt(x -> x).forEach(x -> range(0, dimensions[1]).mapToObj(y -> y)
        .sorted(RefComparator.comparing(y -> y.hashCode())).mapToInt(y -> y).forEach(y -> {
          int row = topology.getIndexFromCoords(x, y);
          if (marks[row] == 0) {
            final int thisIsland = islandNumber.incrementAndGet();
            marks[row] = thisIsland;
            _markIslands(topology, extract, test, marks, maxRecursion, thisIsland, x, y);
          }
        }));
    return marks;
  }

  public static int[] removeTinyInclusions(@Nonnull int[] pixelMap, @Nonnull SparseMatrixFloat graph, int sizeThreshold) {
    return removeTinyInclusions(pixelMap, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(@Nonnull int[] pixelMap, @Nonnull SparseMatrixFloat graph, int smallSize, int largeSize) {
    return removeTinyInclusions(valueCountMap(pixelMap), graph, smallSize, largeSize);
  }

  public static int[] removeTinyInclusions(@Nonnull RefMap<Integer, Long> islandSizes, @Nonnull SparseMatrixFloat graph,
                                           int sizeThreshold) {
    return removeTinyInclusions(islandSizes, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(@Nonnull RefMap<Integer, Long> islandSizes, @Nonnull SparseMatrixFloat graph, int smallSize,
                                           int largeSize) {
    return range(0, graph.rows).map(row -> {
      final int[] cols = graph.getCols(row);
      if (islandSizes.getOrDefault(row, 0l) < smallSize) {
        final int[] largeNeighbors = RefArrays.stream(cols).filter(j -> {
          return islandSizes.getOrDefault(j, 0l) >= largeSize;
        }).toArray();
        if (largeNeighbors.length == 1) {
          return largeNeighbors[0];
        }
      }
      return row;
    }).toArray();
  }

  @Nonnull
  public static BufferedImage flattenColors(@Nonnull Tensor content, RasterTopology topology, @Nonnull RasterAffinity affinity, int n,
                                            @Nonnull SmoothSolver solver) {
    final RefOperator<Tensor> refOperator = solver.solve(topology,
        affinity.wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5)), 1e-4);
    final Tensor tensor = refOperator.iterate(n, content.addRef());
    refOperator.freeRef();
    final BufferedImage image = tensor.toRgbImage();
    tensor.freeRef();
    return image;
  }

  @Nonnull
  public static BufferedImage paintWithRandomColors(@Nonnull RasterTopology topology, int[] pixelMap, @Nonnull SparseMatrixFloat graph) {
    return paint(topology, pixelMap, randomColors(graph, 0));
  }

  @Nonnull
  public static BufferedImage paint(@Nonnull RasterTopology topology, int[] pixelMap, @Nonnull Map<Integer, double[]> colors) {
    final Tensor tensor = new Tensor(topology.getDimensions()).mapCoords(c -> {
      final int[] coords = c.getCoords();
      final int regionId = pixelMap[topology.getIndexFromCoords(coords[0], coords[1])];
      final double[] color = colors.get(regionId);
      return null == color ? 0 : color[coords[2]];
    });
    final BufferedImage image = tensor.toImage();
    tensor.freeRef();
    return image;
  }

  protected static <T> void _markIslands(@Nonnull RasterTopology topology, @Nonnull Function<int[], T> extract, @Nonnull BiPredicate<T, T> test,
                                         int[] marks, int maxRecursion, int indexNumber, int... coords) {
    final int row = topology.getIndexFromCoords(coords[0], coords[1]);
    assert 0 < indexNumber;
    final T rowColor = extract.apply(coords);
    final RefList<int[]> connectivity = topology.connectivity();
    if (maxRecursion > 0) {
      RefArrays.stream(connectivity.get(row)).forEach(col -> {
        if (0 == marks[col]) {
          final int[] toCoords = topology.getCoordsFromIndex(col);
          if (test.test(rowColor, extract.apply(toCoords))) {
            if (0 == marks[col]) {
              marks[col] = indexNumber;
              _markIslands(topology, extract, test, marks, maxRecursion - 1, indexNumber, toCoords);
            }
          }
        }
      });
    }
  }

  private static RefMap<Integer, double[]> randomColors(@Nonnull SparseMatrixFloat graph, int iterations) {
    return randomColors(graph,
        x -> RefDoubleStream.generate(() -> FastRandom.INSTANCE.random() * 255).limit(3).toArray(),
        new RefConcurrentHashMap<>(), iterations);
  }

  private static RefMap<Integer, double[]> randomColors(@Nonnull SparseMatrixFloat graph, @Nonnull Function<Integer, double[]> seedColor,
                                                        @Nonnull RefMap<Integer, double[]> colors, int n) {
    if (n <= 0) {
      if (colors.isEmpty()) {
        return RefArrays.stream(graph.activeRows()).mapToObj(x -> x).collect(RefCollectors.toMap(x -> x, seedColor));
      } else {
        return colors;
      }
    }
    return randomColors(graph, seedColor, iterateColors(graph, seedColor, colors), n - 1);
  }

  private static RefMap<Integer, double[]> iterateColors(@Nonnull SparseMatrixFloat graph, @Nonnull Function<Integer, double[]> seedColor,
                                                         @Nonnull RefMap<Integer, double[]> colors) {
    return RefArrays.stream(graph.activeRows()).parallel().mapToObj(x -> x)
        .collect(RefCollectors.toMap(key -> key, key -> {
          final int[] cols = graph.getCols(key);
          final float[] vals = graph.getVals(key);
          final RefList<double[]> neighborColors = range(0, cols.length)
              .mapToObj(
                  ni -> RefArrays.stream(colors.computeIfAbsent(cols[ni], seedColor)).map(x -> x / vals[ni]).toArray())
              .collect(RefCollectors.toList());
          if (neighborColors.isEmpty())
            return colors.computeIfAbsent(key, seedColor);
          final double[] average = range(0, 3)
              .mapToDouble(i -> -neighborColors.stream().mapToDouble(j -> j[i] - 127).average().orElse(0.0)).toArray();
          final double rms = Math.sqrt(RefArrays.stream(average).map(x -> x * x).average().getAsDouble());
          return RefArrays.stream(average).map(x -> Math.min(Math.max(x / rms * 64 + 127, 0), 255)).toArray();
        }));
  }

}
