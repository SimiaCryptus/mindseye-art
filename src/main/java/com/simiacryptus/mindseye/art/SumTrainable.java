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
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;

public class SumTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(SumTrainable.class);

  private final Trainable[] inner;

  public SumTrainable(Trainable... inner) {
    this.inner = inner;
  }

  public Trainable[] getInner() {
    return RefUtil.addRef(inner);
  }

  @Override
  public Layer getLayer() {
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    pipelineNetwork.add(new SumInputsLayer(), RefArrays.stream(getInner())
        .map(trainable -> trainable.getLayer())
        .map(node -> pipelineNetwork.add(node, pipelineNetwork.getInput(0)))
        .toArray(i->new DAGNode[i])
    ).freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    RefList<PointSample> results = RefArrays.stream(getInner()).map(pointSample -> {
      PointSample measure = pointSample.measure(monitor);
      pointSample.freeRef();
      return measure;
    }).collect(RefCollectors.toList());
    DeltaSet<UUID> delta = RefUtil.get(results.stream().map(x -> {
      DeltaSet<UUID> uuidDeltaSet = x.delta.addRef();
      x.freeRef();
      return uuidDeltaSet;
    }).reduce((a, b) -> {
      a.addInPlace(b);
      return a;
    }));
    StateSet<UUID> weights = RefUtil.get(results.stream().map(x -> {
      StateSet<UUID> uuidStateSet = x.weights.addRef();
      x.freeRef();
      return uuidStateSet;
    }).reduce((a, b) -> {
      return StateSet.union(a, b);
    }));
    double mean = results.stream().mapToDouble(x -> {
      double xMean = x.getMean();
      x.freeRef();
      return xMean;
    }).sum();
    double rate = results.stream().mapToDouble(x -> {
      double xRate = x.getRate();
      x.freeRef();
      return xRate;
    }).average().getAsDouble();
    int sum = results.stream().mapToInt(x -> {
      int count = x.count;
      x.freeRef();
      return count;
    }).sum();
    results.freeRef();
    return new PointSample(delta, weights, mean, rate, sum);
  }

  public void _free() {
    RefUtil.freeRef(inner);
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumTrainable addRef() {
    return (SumTrainable) super.addRef();
  }
}
