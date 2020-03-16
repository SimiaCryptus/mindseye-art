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
import com.simiacryptus.mindseye.art.photo.cuda.RefUnaryOperator;
import com.simiacryptus.mindseye.art.photo.cuda.SparseMatrixFloat;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.util.FastRandom;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

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
    final Map<Integer, Long> countMap = valueCountMap(ints);
    return range(0, RefArrays.stream(ints).max().getAsInt() + 1)
        .map((int i) -> (int) (long) countMap.getOrDefault(i, 0l)).toArray();
  }

  public static Map<Integer, Long> valueCountMap(@Nonnull int[] ints) {
    return Arrays.stream(ints).mapToObj(x -> x)
        .collect(Collectors.groupingBy((Integer x) -> x, Collectors.counting()));
  }

  public static void printHistogram(@Nonnull int[] islands) {
    RefArrays.stream(islands).mapToObj(x -> x).collect(Collectors.groupingBy(x -> x, Collectors.counting()))
        .values().stream().collect(Collectors.groupingBy(x -> x, Collectors.counting())).entrySet().stream()
        .sorted(Comparator.comparingDouble(entry -> {
          long v = -entry.getValue() * entry.getKey();
          RefUtil.freeRef(entry);
          return v;
        }))
        .map(entry -> {
          String msg = String.format("%d regions of size %s", entry.getValue(), entry.getKey());
          RefUtil.freeRef(entry);
          return msg;
        })
        .forEach(x1 -> System.out.println(x1));
  }

  @Nonnull
  public static <T> int[] markIslands(@Nonnull RasterTopology topology, @Nonnull Function<int[], T> extract, @Nonnull BiPredicate<T, T> test,
                                      int maxRecursion, int rows) {
    int[] marks = new int[rows];
    AtomicInteger islandNumber = new AtomicInteger(0);
    int[] dimensions = topology.getDimensions();
    IntStream stream = range(0, dimensions[0]);
    if (!CoreSettings.INSTANCE().singleThreaded) stream = stream.parallel();
    stream.mapToObj(x -> x).sorted(Comparator.comparingInt(x -> x.hashCode()))
        .mapToInt(x -> x).forEach(x -> range(0, dimensions[1]).mapToObj(y -> y)
        .sorted(Comparator.comparing(y -> y.hashCode())).mapToInt(y -> y).forEach(y -> {
          int row = topology.getIndexFromCoords(x, y);
          if (marks[row] == 0) {
            final int thisIsland = islandNumber.incrementAndGet();
            marks[row] = thisIsland;
            _markIslands(topology.addRef(), extract, test, marks, maxRecursion, thisIsland, x, y);
          }
        }));
    topology.freeRef();
    return marks;
  }

  public static int[] removeTinyInclusions(@Nonnull int[] pixelMap, @Nonnull SparseMatrixFloat graph, int sizeThreshold) {
    return removeTinyInclusions(pixelMap, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(@Nonnull int[] pixelMap, @Nonnull SparseMatrixFloat graph, int smallSize, int largeSize) {
    return removeTinyInclusions(valueCountMap(pixelMap), graph, smallSize, largeSize);
  }

  public static int[] removeTinyInclusions(@Nonnull Map<Integer, Long> islandSizes, @Nonnull SparseMatrixFloat graph,
                                           int sizeThreshold) {
    return removeTinyInclusions(islandSizes, graph, sizeThreshold, sizeThreshold);
  }

  public static int[] removeTinyInclusions(@Nonnull Map<Integer, Long> islandSizes, @Nonnull SparseMatrixFloat graph, int smallSize,
                                           int largeSize) {
    return range(0, graph.rows).map(row -> {
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

  @Nonnull
  public static BufferedImage flattenColors(@Nonnull Tensor content, RasterTopology topology, @RefAware @Nonnull RasterAffinity affinity, int n,
                                            @Nonnull SmoothSolver solver) {
    final RefUnaryOperator<Tensor> refUnaryOperator = solver.solve(topology,
        affinity.wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5)), 1e-4);
    RefUtil.freeRef(affinity);
    final Tensor tensor = refUnaryOperator.iterate(n, content);
    refUnaryOperator.freeRef();
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
    Tensor blank = new Tensor(topology.getDimensions());
    final Tensor tensor = blank.mapCoords(c -> {
      final int[] coords = c.getCoords();
      final int regionId = pixelMap[topology.getIndexFromCoords(coords[0], coords[1])];
      final double[] color = colors.get(regionId);
      return null == color ? 0 : color[coords[2]];
    });
    final BufferedImage image = tensor.toImage();
    blank.freeRef();
    tensor.freeRef();
    topology.freeRef();
    return image;
  }

  protected static <T> void _markIslands(@Nonnull RasterTopology topology, @Nonnull Function<int[], T> extract, @Nonnull BiPredicate<T, T> test,
                                         int[] marks, int maxRecursion, int indexNumber, int... coords) {
    final int row = topology.getIndexFromCoords(coords[0], coords[1]);
    assert 0 < indexNumber;
    final T rowColor = extract.apply(coords);
    final List<int[]> connectivity = topology.connectivity();
    if (maxRecursion > 0) {
      Arrays.stream(connectivity.get(row)).forEach(col -> {
        if (0 == marks[col]) {
          final int[] toCoords = topology.getCoordsFromIndex(col);
          if (test.test(rowColor, extract.apply(toCoords))) {
            if (0 == marks[col]) {
              marks[col] = indexNumber;
              _markIslands(topology.addRef(), extract, test, marks, maxRecursion - 1, indexNumber, toCoords);
            }
          }
        }
      });
    }
    topology.freeRef();
  }

  private static Map<Integer, double[]> randomColors(@Nonnull SparseMatrixFloat graph, int iterations) {
    return randomColors(graph,
        x -> DoubleStream.generate(() -> FastRandom.INSTANCE.random() * 255).limit(3).toArray(),
        new ConcurrentHashMap<>(), iterations);
  }

  private static Map<Integer, double[]> randomColors(@Nonnull SparseMatrixFloat graph, @Nonnull Function<Integer, double[]> seedColor,
                                                     @Nonnull Map<Integer, double[]> colors, int n) {
    if (n <= 0) {
      if (colors.isEmpty()) {
        return Arrays.stream(graph.activeRows()).mapToObj(x -> x).collect(Collectors.toMap(x -> x, seedColor));
      } else {
        return colors;
      }
    }
    return randomColors(graph, seedColor, iterateColors(graph, seedColor, colors), n - 1);
  }

  private static Map<Integer, double[]> iterateColors(@Nonnull SparseMatrixFloat graph, @Nonnull Function<Integer, double[]> seedColor,
                                                      @Nonnull Map<Integer, double[]> colors) {
    IntStream stream = Arrays.stream(graph.activeRows());
    if (!CoreSettings.INSTANCE().singleThreaded) stream = stream.parallel();
    return stream.mapToObj(x -> x)
        .collect(Collectors.toMap(key -> key, key -> {
          final int[] cols = graph.getCols(key);
          final float[] vals = graph.getVals(key);
          final List<double[]> neighborColors = range(0, cols.length)
              .mapToObj(
                  ni -> Arrays.stream(colors.computeIfAbsent(cols[ni], seedColor)).map(x -> x / vals[ni]).toArray())
              .collect(Collectors.toList());
          if (neighborColors.isEmpty())
            return colors.computeIfAbsent(key, seedColor);
          final double[] average = range(0, 3)
              .mapToDouble(i -> -neighborColors.stream().mapToDouble(j -> j[i] - 127).average().orElse(0.0)).toArray();
          final double rms = Math.sqrt(Arrays.stream(average).map(x -> x * x).average().getAsDouble());
          return Arrays.stream(average).map(x -> Math.min(Math.max(x / rms * 64 + 127, 0), 255)).toArray();
        }));
  }

}
