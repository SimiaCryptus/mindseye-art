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
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefSystem;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Math.log;

public abstract class RegionAssembler implements Comparator<RegionAssembler.Connection> {

  public final HashSet<Region> regions = new HashSet<>();
  private final int pixels;
  private final int regionCount;
  private final Predicate<Connection> connectionFilter;

  public RegionAssembler(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Map<Integer, Integer> assignments,
                         @Nonnull @RefAware IntFunction<double[]> pixelFunction, @Nonnull @RefAware IntFunction<double[]> coordFunction,
                         Predicate<Connection> connectionFilter) {
    graph.assertSymmetric();
    this.regionCount = graph.rows;
    this.pixels = pixelMap.length;
    this.connectionFilter = connectionFilter;
    final Region[] rawRegions = IntStream.range(0, graph.rows).mapToObj(row -> new Region(row))
        .toArray(i -> new Region[i]);
    final Map<Integer, List<Integer>> pixelAssignmentMap = IntStream.range(0, pixelMap.length).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> pixelMap[x], Collectors.toList()));
    IntStream stream = Arrays.stream(graph.activeRows());
    if (!CoreSettings.INSTANCE().singleThreaded) stream = stream.parallel();
    final List<Region> collect = stream.mapToObj(row -> {
      assert this.regionCount > row;
      final int[] cols = graph.getCols(row);
      final float[] vals = graph.getVals(row);
      assert cols.length == vals.length;
      final Region region = rawRegions[row];
      final Integer assignment = assignments.get(row);
      if (null != assignment)
        region.marks.add(assignment);
      if (cols.length != 0) {
        region.connections_addAll(IntStream.range(0, cols.length).filter(x -> Double.isFinite(vals[x]) && vals[x] != 0)
            .mapToObj(x -> new Connection(region, rawRegions[cols[x]], vals[x])).filter(x -> x.to != x.from)
            .collect(Collectors.toList()));
      }
      final List<Integer> pixels = pixelAssignmentMap.get(row);
      if (pixels != null) {
        pixels.stream().map(value1 -> pixelFunction.apply(value1)).collect(Collectors.toList()).forEach(doubles1 -> region.colorStats.accept(doubles1));
        pixels.stream().map(value -> coordFunction.apply(value)).collect(Collectors.toList()).forEach(doubles -> region.spacialStats.accept(doubles));
        region.pixels.addAll(pixels);
      }
      region.original_regions.add(row);
      return region;
    }).collect(Collectors.toList());
    RefUtil.freeRef(pixelFunction);
    RefUtil.freeRef(coordFunction);
    this.regions.addAll(collect);
    assert this.regions.stream().flatMap(x -> x.connections.values().stream()).allMatch(connection -> {
      final Connection reciprical = connection.reciprical();
      if (null == reciprical)
        return false;
      return Math.abs(connection.value - reciprical.value) < 1e-4;
    });
  }

  @Nonnull
  public int[] getPixelMap() {
    final int[] ints = new int[pixels];
    regions.forEach(i -> {
      final int id = i.minId();
      i.pixels.forEach(p -> ints[p] = id);
    });
    return ints;
  }

  @Nonnull
  public int[] getProjection() {
    final int[] ints = new int[regionCount];
    regions.forEach(i -> {
      final int id = i.minId();
      i.original_regions.forEach(p -> ints[p] = id);
    });
    return ints;
  }

  @Nonnull
  public RegionTree getTree() {
    return new RegionTree(regions.stream().map(x -> x.tree).toArray(i -> new RegionTree[i]));
  }

  @Nonnull
  public static RegionAssembler wrap(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Function<Connection, Double> extractor,
                                     @Nonnull final Tensor content, @Nonnull @RefAware final RasterTopology topology, @Nonnull final Map<Integer, Integer> assignments) {
    return new RegionAssembler(
        graph,
        pixelMap,
        assignments,
        RefUtil.wrapInterface(p -> content.getPixel(topology.getCoordsFromIndex(p)), content, RefUtil.addRef(topology)),
        RefUtil.wrapInterface(p -> Arrays.stream(topology.getCoordsFromIndex(p)).mapToDouble(x -> x).toArray(), topology),
        connection -> connection.to != connection.from && extractor.apply(connection) < Double.POSITIVE_INFINITY
    ) {
      @Override
      public int compare(Connection o1, Connection o2) {
        return Comparator.comparing(extractor).compare(o1, o2);
      }
    };
  }

  public @Nonnull
  static RegionAssembler volumeEntropy(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Tensor content,
                                       @Nonnull RasterTopology topology) {
    return wrap(graph, pixelMap, new Function<Connection, Double>() {
      @Override
      public Double apply(@Nullable Connection entry) {
        if (null == entry)
          return Double.POSITIVE_INFINITY;
        if (0 == entry.to.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;
        if (0 == entry.from.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;
        final double entropy = reduce(-log(entry.value / entry.to.getConnectionWeight()),
            -log(entry.value / entry.from.getConnectionWeight()));
        final double smallness = reduce(entry.to.pixels.size(), entry.from.pixels.size());
        return smallness * entropy;
      }

      double reduce(double a, double b) {
        return Math.min(a, b);
      }
    }, content, topology, new HashMap<Integer, Integer>());
  }

  public @Nonnull
  static RegionAssembler simpleEntropy(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Tensor content,
                                       @Nonnull RasterTopology topology) {
    return wrap(graph, pixelMap, new Function<Connection, Double>() {
      @Override
      public Double apply(@Nullable Connection entry) {
        if (null == entry)
          return Double.POSITIVE_INFINITY;
        if (0 == entry.to.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;
        if (0 == entry.from.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;
        return reduce(-log(entry.value / entry.to.getConnectionWeight()),
            -log(entry.value / entry.from.getConnectionWeight()));
      }

      double reduce(double a, double b) {
        return Math.min(a, b);
      }
    }, content, topology, new HashMap<Integer, Integer>());
  }

  public @Nonnull
  static RegionAssembler epidemic(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Tensor content,
                                  @Nonnull RasterTopology topology, @Nonnull Map<Integer, Integer> assignments) {
    return wrap(graph, pixelMap, new Function<Connection, Double>() {
      @Override
      public Double apply(@Nullable Connection entry) {
        if (null == entry)
          return Double.POSITIVE_INFINITY;
        if (0 == entry.to.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;
        if (0 == entry.from.getConnectionWeight())
          return -Double.POSITIVE_INFINITY;

        final int[] to_marks = entry.to.marks.stream().mapToInt(x -> x).limit(1).toArray();
        final int[] from_marks = entry.from.marks.stream().mapToInt(x -> x).limit(1).toArray();
        if (to_marks.length == 0 && from_marks.length == 0)
          return Double.POSITIVE_INFINITY;
        if (to_marks.length > 0 && from_marks.length > 0) {
          if (to_marks[0] == from_marks[0])
            return -Double.POSITIVE_INFINITY;
          else
            return Double.POSITIVE_INFINITY;
        }

        final double entropy = reduce(-log(entry.value / entry.to.getConnectionWeight()),
            -log(entry.value / entry.from.getConnectionWeight()));
        final double smallness = reduce(entry.to.pixels.size(), entry.from.pixels.size());
        return smallness * entropy;
      }

      double reduce(double a, double b) {
        return Math.min(a, b);
      }
    }, content, topology, assignments);
  }

  public @Nonnull
  static RegionAssembler volume5D(@Nonnull SparseMatrixFloat graph, @Nonnull int[] pixelMap, @Nonnull Tensor content,
                                  @Nonnull RasterTopology topology) {
    final double minVol = 5e-1;
    final double color_coeff = 1e3;
    return wrap(graph, pixelMap, new Function<Connection, Double>() {
      @Override
      public Double apply(@Nullable Connection entry) {
        if (null == entry)
          return Double.POSITIVE_INFINITY;
        final int to_pixels = entry.to.pixels.size();
        final int from_pixels = entry.from.pixels.size();
        final double log_vol_to = logVol(entry.to.colorStats, entry.to.spacialStats);
        final double log_vol_from = logVol(entry.from.colorStats, entry.from.spacialStats);
        final double log_vol_union = logVol(union(entry.to.colorStats, entry.from.colorStats),
            union(entry.to.spacialStats, entry.from.spacialStats));
        final double sourceVolume = log_vol_to * to_pixels + log_vol_from * from_pixels;
        final double resultVolume = log_vol_union * (to_pixels + from_pixels);
        final double split_entropy = to_pixels * log((double) to_pixels / (to_pixels + from_pixels))
            + from_pixels * log((double) from_pixels / (to_pixels + from_pixels));
        return (resultVolume - sourceVolume) / split_entropy;
      }

      public double logVol(@Nonnull DoubleVectorStatistics colorStats, @Nonnull DoubleVectorStatistics spacialStats) {
        return color_coeff * log(volumeStdDev(minVol, colorStats)) + log(volumeExtrema(minVol, spacialStats));
      }

      public double volumeExtrema(double minVol, @Nonnull DoubleVectorStatistics union) {
        return Arrays.stream(union.firstOrder).mapToDouble(statistics -> statistics.getMax() - statistics.getMin())
            .map(x -> Math.abs(x) < minVol ? minVol : x).reduce((a, b) -> a * b).orElse(0);
      }

      public double volumeStdDev(double minVol, @Nonnull DoubleVectorStatistics stats) {
        final DoubleSummaryStatistics[] firstOrder = stats.firstOrder;
        final DoubleSummaryStatistics[] secondOrder = stats.secondOrder;
        return IntStream.range(0, firstOrder.length)
            .mapToDouble(
                i -> Math.pow(Math.abs(secondOrder[i].getAverage() - Math.pow(firstOrder[i].getAverage(), 2)), 0.5))
            .map(x -> Math.abs(x) < minVol ? minVol : x).reduce((a, b) -> a * b).orElse(0);
      }

      @Nonnull
      private DoubleVectorStatistics union(@Nonnull DoubleVectorStatistics a, @Nonnull DoubleVectorStatistics b) {
        final DoubleVectorStatistics statistics = new DoubleVectorStatistics(a.firstOrder.length);
        statistics.combine(a);
        statistics.combine(b);
        return statistics;
      }
    }, content, topology, new HashMap<Integer, Integer>());
  }

  @Nonnull
  public static int[] reduce(@Nonnull SparseMatrixFloat graph, int targetCount, @Nonnull final int[] sizes, @Nonnull Tensor content,
                             @Nonnull RasterTopology topology) {
    return wrap(graph, sizes, (Connection entry) -> {
      return null == entry ? Double.POSITIVE_INFINITY : entry.value;
    }, content, topology, new HashMap<Integer, Integer>()).reduceTo(targetCount).getProjection();
  }

  private static void assertEmpty(@Nonnull List<?> collect) {
    if (!collect.isEmpty()) {
      throw new IllegalArgumentException("Items: " + collect.size());
    }
  }

  @Nonnull
  public RegionAssembler reduceTo(int count) {
    while (regions.parallelStream().map(region1 -> region1.connections_stream())
        .filter(stream -> stream.filter(connectionFilter).count() > 0).limit(count + 1).count() > count) {
      final int limit = Math.max(1, (regions.size() - count) / 100);
      List<Connection> first = regions.parallelStream().flatMap(region -> region.connections_stream_parallel())
          .filter(connectionFilter).sorted(this).limit(limit).collect(Collectors.toList());
      if (first.isEmpty()) {
        System.out.println("No connections left");
        break;
      } else {
        final HashSet<Region> touched = new HashSet<>();
        first.stream().filter(connection -> touched.add(connection.to) && touched.add(connection.from))
            .forEach(connection1 -> connection1.join());
      }
    }
    return this;
  }

  public static class RegionTree {
    public final int[] regions;
    @Nonnull
    public final RegionTree[] children;

    public RegionTree(int... regions) {
      this.regions = regions;
      children = new RegionTree[]{};
    }

    public RegionTree(@Nonnull RegionTree... children) {
      this.regions = Arrays.stream(children).flatMapToInt(x -> Arrays.stream(x.regions)).toArray();
      this.children = children;
    }
  }

  public class Connection {
    public final Region from;
    public final Region to;
    public final float value;

    public Connection(Region from, Region to, float value) {
      this.from = from;
      this.to = to;
      this.value = value;
      assert value > 0.0;
    }

    public Connection reciprical() {
      return to.connections.get(from);
    }

    protected void join() {
      final Region toRemove;
      final Region consolidated;
      final int to_id = this.to.minId();
      final int from_id = this.from.minId();
      if (to_id < from_id) {
        toRemove = this.from;
        consolidated = this.to;
      } else {
        toRemove = this.to;
        consolidated = this.from;
      }
      if (!regions.remove(toRemove)) {
        System.out.println("Remove dead connection to Region " + toRemove.minId());
        if (!this.from.connections_remove(this)) {
          throw new IllegalStateException();
        } else {
          return;
        }
      } else {
        //com.simiacryptus.ref.wrappers.System.out.println("Remove Region " + minId(toRemove));
      }
      consolidated.union(toRemove);
      toRemove.clear();
    }
  }

  public class Region {
    public final HashMap<Region, Connection> connections = new HashMap<>();
    public final HashSet<Integer> pixels = new HashSet<>();
    public final HashSet<Integer> marks = new HashSet<>();
    public final HashSet<Integer> original_regions = new HashSet<>();
    public final DoubleVectorStatistics colorStats = new DoubleVectorStatistics(3);
    public final DoubleVectorStatistics spacialStats = new DoubleVectorStatistics(2);
    @Nullable
    public RegionTree tree;
    private double connectionWeight = 0;

    public Region(int id, @Nonnull Connection... connections) {
      this.original_regions.add(id);
      tree = new RegionTree(id);
      Arrays.stream(connections).forEach(connection -> connections_add(connection));
    }

    public double getConnectionWeight() {
      return connectionWeight;
    }

    public Stream<Connection> connections_stream() {
      return connections.values().stream();
    }

    public Stream<Connection> connections_stream_parallel() {
      return connections.values().parallelStream();
    }

    public void connections_clear() {
      connectionWeight = 0;
      connections.clear();
    }

    public boolean connections_add(@Nonnull Connection connection) {
      assert connection.from == this;
      final boolean add = null == connections.put(connection.to, connection);
      if (add)
        connectionWeight += connection.value;
      return add;
    }

    public boolean connections_remove(@Nonnull Connection connection) {
      final boolean remove = null != connections.remove(connection.to);
      if (remove)
        connectionWeight -= connection.value;
      return remove;
    }

    public boolean connections_addAll(@Nonnull Collection<? extends Connection> connections) {
      return connections.stream().filter(x -> !this.connections_add(x)).allMatch(connection -> connections_add(connection));
    }

    public int minId() {
      return original_regions.stream().mapToInt(x -> x).min().getAsInt();
    }

    private void union(@Nonnull Region other) {
      final List<Connection> newConnections = Stream.concat(other.connections_stream(), connections_stream())
          .filter(k -> k.to != other && k.to != this)
          .collect(Collectors.groupingBy(k -> k.to, Collectors.reducing((a, b) -> {
            assert a.from == this || a.from == other;
            assert b.from == this || b.from == other;
            assert a.to == b.to;
            return new Connection(a.from, a.to, a.value + b.value);
          }))).values().stream().map(optional -> RefUtil.get(optional)).map(c -> new Connection(this, c.to, c.value))
          .collect(Collectors.toList());

      this.colorStats.combine(other.colorStats);
      this.spacialStats.combine(other.spacialStats);
      this.pixels.addAll(other.pixels);
      this.original_regions.addAll(other.original_regions);
      this.marks.addAll(other.marks);
      this.tree = new RegionTree(this.tree, other.tree);
      connections_clear();
      connections_addAll(newConnections);

      newConnections.stream().allMatch(v -> {
        final Region thirdNode = v.to;
        final List<Connection> connectionsToRemove = thirdNode.connections_stream()
            .filter(x -> x.to == other || x.to == this).collect(Collectors.toList());
        //assert Math.abs(connectionsToRemove.stream().mapToDouble(x -> x.value).sum() - v.value) < 1e-3;
        assertEmpty(
            connectionsToRemove.stream().filter(o -> !thirdNode.connections_remove(o)).collect(Collectors.toList()));
        thirdNode.connections_add(new Connection(thirdNode, v.from, v.value));
        return true;
      });
    }

    private void clear() {
      this.pixels.clear();
      this.marks.clear();
      this.original_regions.clear();
      this.tree = null;
      connections_clear();
    }
  }

}
