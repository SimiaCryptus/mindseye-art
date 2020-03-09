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

import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefSet;
import com.simiacryptus.ref.wrappers.RefStream;

import javax.annotation.Nonnull;
import java.util.DoubleSummaryStatistics;
import java.util.function.*;
import java.util.stream.Collector;

public class DoubleVectorStatistics implements Consumer<double[]> {

  @Nonnull
  final DoubleSummaryStatistics[] firstOrder;
  @Nonnull
  final DoubleSummaryStatistics[] secondOrder;

  public DoubleVectorStatistics(int length) {
    firstOrder = RefIntStream.range(0, length).mapToObj(i -> new DoubleSummaryStatistics())
        .toArray(i -> new DoubleSummaryStatistics[i]);
    secondOrder = RefIntStream.range(0, length).mapToObj(i -> new DoubleSummaryStatistics())
        .toArray(i -> new DoubleSummaryStatistics[i]);
  }

  @Nonnull
  public static Collector<double[], DoubleVectorStatistics, DoubleVectorStatistics> collector(int dims) {
    return new Collector<double[], DoubleVectorStatistics, DoubleVectorStatistics>() {
      @Nonnull
      @Override
      public Supplier<DoubleVectorStatistics> supplier() {
        return () -> new DoubleVectorStatistics(dims);
      }

      @Nonnull
      @Override
      public BiConsumer<DoubleVectorStatistics, double[]> accumulator() {
        return (a, b) -> a.accept(b);
      }

      @Nonnull
      @Override
      public BinaryOperator<DoubleVectorStatistics> combiner() {
        return (a, b) -> {
          final DoubleVectorStatistics statistics = new DoubleVectorStatistics(a.firstOrder.length);
          statistics.combine(a);
          statistics.combine(b);
          return statistics;
        };
      }

      @Nonnull
      @Override
      public Function<DoubleVectorStatistics, DoubleVectorStatistics> finisher() {
        return x -> x;
      }

      @Override
      public RefSet<Characteristics> characteristics() {
        return RefStream.of(Characteristics.UNORDERED).collect(RefCollectors.toSet());
      }
    };
  }

  @Override
  public void accept(@Nonnull double[] doubles) {
    assert firstOrder.length == doubles.length;
    RefIntStream.range(0, doubles.length).forEach(i -> firstOrder[i].accept(doubles[i]));
    RefIntStream.range(0, doubles.length).forEach(i -> secondOrder[i].accept(doubles[i] * doubles[i]));
  }

  public void combine(@Nonnull DoubleVectorStatistics colorStats) {
    assert firstOrder.length == colorStats.firstOrder.length;
    RefIntStream.range(0, firstOrder.length).forEach(i -> firstOrder[i].combine(colorStats.firstOrder[i]));
    RefIntStream.range(0, secondOrder.length).forEach(i -> secondOrder[i].combine(colorStats.secondOrder[i]));
  }
}
