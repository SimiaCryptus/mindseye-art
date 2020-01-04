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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.wrappers.RefCollectors;

import java.util.concurrent.atomic.AtomicInteger;

public @com.simiacryptus.ref.lang.RefAware
class SearchRadiusTopology extends ContentTopology {
  private int neighborhoodSize = 4;
  private double maxChromaDist = 0.2;
  private double maxSpatialDist = 8;
  private boolean verbose = true;
  private int initialRadius = 1;
  private boolean spatialPriority = false;
  private boolean selfRef = false;

  public SearchRadiusTopology(Tensor content) {
    super(content);
  }

  public int getInitialRadius() {
    return initialRadius;
  }

  public SearchRadiusTopology setInitialRadius(int initialRadius) {
    this.initialRadius = initialRadius;
    return this;
  }

  public double getMaxChromaDist() {
    return maxChromaDist;
  }

  public SearchRadiusTopology setMaxChromaDist(double maxChromaDist) {
    this.maxChromaDist = maxChromaDist;
    return this;
  }

  public double getMaxSpatialDist() {
    return maxSpatialDist;
  }

  public SearchRadiusTopology setMaxSpatialDist(double maxSpatialDist) {
    this.maxSpatialDist = maxSpatialDist;
    return this;
  }

  public int getNeighborhoodSize() {
    return neighborhoodSize;
  }

  public SearchRadiusTopology setNeighborhoodSize(int neighborhoodSize) {
    this.neighborhoodSize = neighborhoodSize;
    return this;
  }

  public boolean isSelfRef() {
    return selfRef;
  }

  public SearchRadiusTopology setSelfRef(boolean selfRef) {
    this.selfRef = selfRef;
    return this;
  }

  public boolean isSpatialPriority() {
    return spatialPriority;
  }

  public SearchRadiusTopology setSpatialPriority(boolean spatialPriority) {
    this.spatialPriority = spatialPriority;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public SearchRadiusTopology setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public static Tensor medianFilter(Tensor content) {
    return medianFilter(content, 1);
  }

  public static Tensor medianFilter(Tensor content, int window) {
    final int[] dimensions = content.getDimensions();
    return content.mapCoords(c -> {
      final int[] pos = c.getCoords();
      final double[] neighbors = com.simiacryptus.ref.wrappers.RefIntStream
          .range(Math.max(0, pos[0] - window), Math.min(dimensions[0], pos[0] + window + 1))
          .mapToObj(x -> com.simiacryptus.ref.wrappers.RefIntStream
              .range(Math.max(0, pos[1] - window), Math.min(dimensions[1], pos[1] + window + 1))
              .mapToDouble(y -> content.get(x, y, pos[2])).toArray())
          .flatMapToDouble(x -> com.simiacryptus.ref.wrappers.RefArrays.stream(x)).sorted().toArray();
      return neighbors[neighbors.length / 2];
    });
  }

  public static @SuppressWarnings("unused")
  SearchRadiusTopology[] addRefs(SearchRadiusTopology[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SearchRadiusTopology::addRef)
        .toArray((x) -> new SearchRadiusTopology[x]);
  }

  public static @SuppressWarnings("unused")
  SearchRadiusTopology[][] addRefs(SearchRadiusTopology[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SearchRadiusTopology::addRefs)
        .toArray((x) -> new SearchRadiusTopology[x][]);
  }

  private static double median(com.simiacryptus.ref.wrappers.RefDoubleStream doubleStream) {
    return doubleStream.average().getAsDouble();
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<int[]> connectivity() {
    final double growth = Math.sqrt(2);
    final com.simiacryptus.ref.wrappers.RefList<int[]> symmetric = dual(
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i -> {
          final int[] pos = getCoordsFromIndex(i);
          final double[] pixel = pixel(i);
          final com.simiacryptus.ref.wrappers.RefArrayList<int[]> neighbors = new com.simiacryptus.ref.wrappers.RefArrayList<>();
          AtomicInteger window = new AtomicInteger(getInitialRadius());
          while (neighbors.stream().mapToInt(x -> x.length).sum() < getNeighborhoodSize()
              && window.get() < (maxSpatialDist > 0 ? Math.min(dimensions[0], maxSpatialDist) : dimensions[0])) {
            final int windowSize = window.get();
            final int windowMin = windowSize > 2 ? (windowSize / 2) : -1;
            final int[] matchingGlobal = com.simiacryptus.ref.wrappers.RefIntStream
                .range(Math.max(0, pos[0] - windowSize), Math.min(dimensions[0], pos[0] + windowSize + 1))
                .mapToObj(x -> com.simiacryptus.ref.wrappers.RefIntStream
                    .range(Math.max(0, pos[1] - windowSize), Math.min(dimensions[1], pos[1] + windowSize + 1))
                    .filter(y -> {
                      final int index = getIndexFromCoords(x, y);
                      if (!isSelfRef() && index == i)
                        return false;
                      final int dx = pos[0] - x;
                      final int dy = pos[1] - y;
                      if (Math.abs(dy) <= windowMin && Math.abs(dx) <= windowMin) {
                        return false;
                      }
                      if (maxSpatialDist > 0) {
                        if (dx * dx + dy * dy > maxSpatialDist * maxSpatialDist) {
                          return false;
                        }
                      }
                      final double dist = chromaDistance(pixel, pixel(index));
                      return dist <= getMaxChromaDist();
                    }).map(y -> getIndexFromCoords(x, y)).toArray())
                .flatMapToInt(x -> com.simiacryptus.ref.wrappers.RefArrays.stream(x)).toArray();
            if (isSpatialPriority()) {
              collectSpacialNeighbors(pos, neighbors, matchingGlobal);
            } else {
              collectChromaNeighbors(pixel, neighbors, matchingGlobal);
            }
            window.set(Math.max((int) (windowSize * growth), windowSize + 1));
          }
          return neighbors.stream().flatMapToInt(com.simiacryptus.ref.wrappers.RefArrays::stream).toArray();
        }).collect(RefCollectors.toList()));
    if (isVerbose())
      log(symmetric);
    return symmetric;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SearchRadiusTopology addRef() {
    return (SearchRadiusTopology) super.addRef();
  }

  private void collectSpacialNeighbors(int[] pos, com.simiacryptus.ref.wrappers.RefArrayList<int[]> neighbors,
                                       int[] matchingGlobal) {
    final com.simiacryptus.ref.wrappers.RefMap<Long, com.simiacryptus.ref.wrappers.RefList<Integer>> collect = com.simiacryptus.ref.wrappers.RefArrays
        .stream(matchingGlobal).mapToObj(x -> x).collect(RefCollectors.groupingBy((Integer x) -> {
          final int[] coords = getCoordsFromIndex(x);
          int dx = coords[0] - pos[0];
          int dy = coords[1] - pos[1];
          return ((long) dx * dx + (long) dy * dy);
        }, RefCollectors.toList()));
    final long[] globalRadii = collect.keySet().stream().mapToLong(x -> x).sorted().toArray();
    for (int ring = 0; ring < globalRadii.length; ring++) {
      if (neighbors.stream().flatMapToInt(com.simiacryptus.ref.wrappers.RefArrays::stream).distinct()
          .count() >= getNeighborhoodSize())
        break;
      neighbors.add(collect.get(globalRadii[ring]).stream().mapToInt(x -> x).toArray());
    }
  }

  private void collectChromaNeighbors(double[] fromPixel, com.simiacryptus.ref.wrappers.RefArrayList<int[]> neighbors,
                                      int[] matchingGlobal) {
    neighbors.add(com.simiacryptus.ref.wrappers.RefArrays.stream(matchingGlobal).mapToObj(x -> x)
        .sorted(com.simiacryptus.ref.wrappers.RefComparator.comparing(x -> chromaDistance(pixel(x), fromPixel)))
        .mapToInt(x -> x).distinct().limit(getNeighborhoodSize() - neighbors.stream().mapToInt(x -> x.length).sum())
        .toArray());
  }
}
