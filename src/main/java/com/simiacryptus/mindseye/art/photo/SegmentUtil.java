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

import com.simiacryptus.mindseye.art.photo.cuda.SparseMatrixFloat;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.util.FastRandom;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

  public static int[] islandSizes(int[] islands) {
    final Map<Integer, Long> islandSizes = Arrays.stream(islands).mapToObj(x -> x).collect(Collectors.groupingBy(x -> x, Collectors.counting()));
    return IntStream.range(0, Arrays.stream(islands).max().getAsInt() + 1).map((int i) -> (int) (long) islandSizes.getOrDefault(i, 0l)).toArray();
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

  public static BufferedImage randomColors(RasterTopology topology, Tensor content, int[] islands) {
    final List<double[]> colors = IntStream.range(0, Arrays.stream(islands).max().getAsInt() + 1)
        .mapToObj(i -> DoubleStream.generate(() -> FastRandom.INSTANCE.random() * 255).limit(3).toArray())
        .collect(Collectors.toList());
    return content.mapCoords(c -> {
      final int[] coords = c.getCoords();
      return colors.get(islands[topology.getIndexFromCoords(coords[0], coords[1])])[coords[2]];
    }).toImage();
  }

  public static <T> int[] markIslands(RasterTopology topology, SparseMatrixFloat laplacian, Function<int[], T> extract, BiPredicate<T, T> test, int maxRecursion) {
    int[] marks = new int[laplacian.rows];
    int islandNumber = 0;
    int[] dimensions = topology.getDimensions();
    for (int x = 0; x < dimensions[0]; x++) {
      for (int y = 0; y < dimensions[1]; y++) {
        islandNumber = markIslands(topology, extract, test, laplacian, marks, islandNumber, maxRecursion, x, y);
      }
    }
    return marks;
  }

  public static <T> int markIslands(RasterTopology topology, Function<int[], T> extract, BiPredicate<T, T> test, SparseMatrixFloat laplacian, int[] marks, int islandNumber, int maxRecursion, int... coords) {
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
    if (maxRecursion > 0) for (int col : laplacian.getCols(row)) {
      if (0 == marks[col]) {
        final int[] toCoords = topology.getCoordsFromIndex(col);
        if (test.test(rowColor, extract.apply(toCoords))) {
          marks[col] = thisIsland;
          islandNumber = markIslands(topology, extract, test, laplacian, marks, islandNumber, maxRecursion - 1, toCoords);
        }
      }
    }
    return islandNumber;
  }

  public static int[] reduceIslands(SparseMatrixFloat refined_graph, int targetCount) {
    final List<Map<Integer, Float>> neighborList = IntStream.range(0, refined_graph.rows).mapToObj(i -> {
      final int[] cols = refined_graph.getCols(i);
      final float[] vals = refined_graph.getVals(i);
      assert cols.length == vals.length;
      if (cols.length == 0) return new HashMap<Integer, Float>();
      return IntStream.range(0, cols.length).mapToObj(x -> x).collect(Collectors.toMap(x -> cols[x], x -> vals[x]));
    }).collect(Collectors.toList());

    final int[] islands_joined = IntStream.range(0, refined_graph.rows).toArray();
    while (neighborList.stream().filter(x -> x.size() > 0).count() > targetCount) {
      final List<Map.Entry<Integer, Float>> nearestNeighbors = IntStream.range(0, neighborList.size()).mapToObj(neighborIndex -> {
        final Map<Integer, Float> map = neighborList.get(neighborIndex);
        return map.entrySet().stream().filter(e -> !e.getKey().equals(neighborIndex)).sorted(Comparator.comparing(e -> -e.getValue())).findFirst().orElse(null);
      }).collect(Collectors.toList());
      final int nearestIndex = IntStream.range(0, nearestNeighbors.size()).mapToObj(x -> x).sorted(Comparator.comparing(x -> {
        final Map.Entry<Integer, Float> entry = nearestNeighbors.get(x);
        return null==entry?Double.POSITIVE_INFINITY:-entry.getValue();
      })).findFirst().get();
      final int nearestTarget = nearestNeighbors.get(nearestIndex).getKey();
      assert nearestTarget != nearestIndex;
      islands_joined[nearestTarget] = islands_joined[nearestIndex] = Math.min(islands_joined[nearestIndex], islands_joined[nearestTarget]);
      join(neighborList, nearestIndex, nearestTarget);
    }
    return islands_joined;
  }

  public static void join(List<Map<Integer, Float>> neighborList, int nearestIndex, int nearestTarget) {
    final Map<Integer, Float> mapA = neighborList.get(nearestIndex);
    final Map<Integer, Float> mapB = neighborList.get(nearestTarget);
    neighborList.set(nearestIndex, new HashMap<>());
    final Map<Integer, Float> newMap = Stream.concat(
        mapA.keySet().stream(),
        mapB.keySet().stream()
    ).distinct().filter(k -> k != nearestIndex && k != nearestTarget).collect(Collectors.toMap(k -> k, k -> mapA.getOrDefault(k, 0.0f) + mapB.getOrDefault(k, 0.0f)));
    neighborList.set(nearestTarget, newMap);
    newMap.forEach((k, v) -> {
      final Map<Integer, Float> targetMap = neighborList.get(k);
      targetMap.remove(nearestIndex);
      targetMap.put(nearestTarget, v);
    });
  }

  public static int[] removeTinyInclusions(int[] islands, SparseMatrixFloat island_graph, int sizeThreshold) {
    final int[] islandSizes = islandSizes(islands);
    final int[] islands_refined = IntStream.range(0, islands.length).toArray();
    for (int i = 0; i < island_graph.rows; i++) {
      final int[] cols = island_graph.getCols(i);
      if (islandSizes[i] < sizeThreshold) {
        final int[] largeNeighbors = Arrays.stream(cols).filter(j -> islandSizes[j] >= sizeThreshold).toArray();
        if (largeNeighbors.length == 1) {
          islands_refined[i] = largeNeighbors[0];
        }
      }
    }
    return islands_refined;
  }
}
