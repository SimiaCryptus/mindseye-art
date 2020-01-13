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

package com.simiacryptus.mindseye.art;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.StateSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.UUID;

public class SumTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(SumTrainable.class);

  private final Trainable[] inner;

  public SumTrainable(Trainable... inner) {
    this.inner = inner;
  }

  public Trainable[] getInner() {
    return inner;
  }

  @NotNull
  @Override
  public Layer getLayer() {
    return null;
  }

  public static @SuppressWarnings("unused") SumTrainable[] addRefs(SumTrainable[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumTrainable::addRef).toArray((x) -> new SumTrainable[x]);
  }

  public static @SuppressWarnings("unused") SumTrainable[][] addRefs(SumTrainable[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumTrainable::addRefs)
        .toArray((x) -> new SumTrainable[x][]);
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    RefList<PointSample> results = RefArrays.stream(getInner()).map(x -> x.measure(monitor))
        .collect(RefCollectors.toList());
    DeltaSet<UUID> delta = RefUtil.get(results.stream().map(x -> x.delta.addRef()).reduce((a, b) -> {
      DeltaSet<UUID> c = a.addInPlace(b);
      b.freeRef();
      return c;
    }));
    StateSet<UUID> weights = RefUtil.get(results.stream().map(x -> x.weights.addRef()).reduce((a, b) -> {
      StateSet<UUID> c = StateSet.union(a, b);
      a.freeRef();
      b.freeRef();
      return c;
    }));
    double mean = results.stream().mapToDouble(x -> x.getMean()).sum();
    double rate = results.stream().mapToDouble(x -> x.getRate()).average().getAsDouble();
    int sum = results.stream().mapToInt(x -> x.count).sum();
    results.forEach(ReferenceCountingBase::freeRef);
    final PointSample pointSample = new PointSample(delta, weights, mean, rate, sum);
    delta.freeRef();
    weights.freeRef();
    return pointSample;
  }

  public void _free() {
    if (null != getInner())
      RefArrays.stream(getInner()).forEach(ReferenceCounting::freeRef);
    super._free();
  }

  public @Override @SuppressWarnings("unused") SumTrainable addRef() {
    return (SumTrainable) super.addRef();
  }
}
