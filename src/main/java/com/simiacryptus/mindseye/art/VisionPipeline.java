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

import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

public final class VisionPipeline extends ReferenceCountingBase {
  private static final Logger logger = LoggerFactory.getLogger(VisionPipeline.class);

  public final String name;

  private final RefLinkedHashMap<String, PipelineNetwork> networks = new RefLinkedHashMap<>();

  private final RefArrayList<VisionPipelineLayer> layers = new RefArrayList<>();

  public VisionPipeline(String name, @Nonnull @RefAware VisionPipelineLayer... values) {
    this(name, RefArrays.asList(values));
  }

  public VisionPipeline(String name, @Nonnull @RefAware RefCollection<VisionPipelineLayer> values) {
    this.name = name;
    final PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    try {
      values.forEach(value -> {
        pipelineNetwork.add(value.getLayer()).freeRef();
        PipelineNetwork layer = pipelineNetwork.copyPipeline();
        layer.freeze();
        RefUtil.freeRef(networks.put(value.name(), layer));
        layers.add(value);
      });
    } catch (Throwable e) {
      logger.warn("Error", e);
      throw new RuntimeException(e);
    } finally {
      pipelineNetwork.freeRef();
      values.freeRef();
    }
  }

  @Nonnull
  public RefList<VisionPipelineLayer> getLayerList() {
    return new RefArrayList<>(layers.addRef());
  }

  @Nonnull
  public RefLinkedHashMap<String, PipelineNetwork> getNetworks() {
    assertAlive();
    return new RefLinkedHashMap<>(networks.addRef());
  }

  public void _free() {
    super._free();
    layers.freeRef();
    networks.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  VisionPipeline addRef() {
    return (VisionPipeline) super.addRef();
  }

  public PipelineNetwork get(String name) {
    return networks.get(name);
  }
}
