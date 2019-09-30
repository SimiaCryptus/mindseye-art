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

import java.util.DoubleSummaryStatistics;
import java.util.Set;
import java.util.function.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DoubleVectorStatistics implements Consumer<double[]> {

  final DoubleSummaryStatistics[] firstOrder;
  final DoubleSummaryStatistics[] secondOrder;
  public DoubleVectorStatistics(int length) {
    firstOrder = IntStream.range(0, length).mapToObj(i -> new DoubleSummaryStatistics()).toArray(i -> new DoubleSummaryStatistics[i]);
    secondOrder = IntStream.range(0, length).mapToObj(i -> new DoubleSummaryStatistics()).toArray(i -> new DoubleSummaryStatistics[i]);
  }

  public static Collector<double[], DoubleVectorStatistics, DoubleVectorStatistics> collector(int dims) {
    return new Collector<double[], DoubleVectorStatistics, DoubleVectorStatistics>() {
      @Override
      public Supplier<DoubleVectorStatistics> supplier() {
        return () -> new DoubleVectorStatistics(dims);
      }

      @Override
      public BiConsumer<DoubleVectorStatistics, double[]> accumulator() {
        return (a, b) -> a.accept(b);
      }

      @Override
      public BinaryOperator<DoubleVectorStatistics> combiner() {
        return (a, b) -> {
          final DoubleVectorStatistics statistics = new DoubleVectorStatistics(a.firstOrder.length);
          statistics.combine(a);
          statistics.combine(b);
          return statistics;
        };
      }

      @Override
      public Function<DoubleVectorStatistics, DoubleVectorStatistics> finisher() {
        return x -> x;
      }

      @Override
      public Set<Characteristics> characteristics() {
        return Stream.of(
            Characteristics.UNORDERED
        ).collect(Collectors.toSet());
      }
    };
  }

  @Override
  public void accept(double[] doubles) {
    assert firstOrder.length == doubles.length;
    IntStream.range(0, doubles.length).forEach(i -> firstOrder[i].accept(doubles[i]));
    IntStream.range(0, doubles.length).forEach(i -> secondOrder[i].accept(doubles[i] * doubles[i]));
  }

  public void combine(DoubleVectorStatistics colorStats) {
    assert firstOrder.length == colorStats.firstOrder.length;
    IntStream.range(0, firstOrder.length).forEach(i -> firstOrder[i].combine(colorStats.firstOrder[i]));
    IntStream.range(0, secondOrder.length).forEach(i -> secondOrder[i].combine(colorStats.secondOrder[i]));
  }
}
