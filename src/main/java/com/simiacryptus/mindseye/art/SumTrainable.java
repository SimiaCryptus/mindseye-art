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

import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.StateSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class SumTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(SumTrainable.class);

  private final Trainable[] inner;

  public SumTrainable(Trainable... inner) {
    this.inner = inner;
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    List<PointSample> results = Arrays.stream(inner).map(x -> x.measure(monitor)).collect(Collectors.toList());
    DeltaSet<UUID> delta = results.stream().map(x -> x.delta.addRef()).reduce((a, b) -> {
      DeltaSet<UUID> c = a.addInPlace(b);
      b.freeRef();
      return c;
    }).get();
    StateSet<UUID> weights = results.stream().map(x -> x.weights.addRef()).reduce((a, b) -> {
      StateSet<UUID> c = StateSet.union(a, b);
      a.freeRef();
      b.freeRef();
      return c;
    }).get();
    double mean = results.stream().mapToDouble(x -> x.getMean()).sum();
    double rate = results.stream().mapToDouble(x -> x.getRate()).average().getAsDouble();
    int sum = results.stream().mapToInt(x -> x.count).sum();
    results.forEach(ReferenceCountingBase::freeRef);
    return new PointSample(delta, weights, mean, rate, sum);
  }

  @Override
  public Layer getLayer() {
    return null;
  }

  @Override
  protected void _free() {
    if (null != inner) Arrays.stream(inner).forEach(ReferenceCounting::freeRef);
    super._free();
  }
}
