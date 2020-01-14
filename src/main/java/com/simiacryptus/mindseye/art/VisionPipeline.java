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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefLinkedHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public class VisionPipeline<T extends VisionPipelineLayer> extends ReferenceCountingBase {
  private static final Logger logger = LoggerFactory.getLogger(VisionPipeline.class);

  public final String name;

  private final RefLinkedHashMap<T, PipelineNetwork> layers = new RefLinkedHashMap<>();

  public VisionPipeline(String name, @Nonnull T... values) {
    this.name = name;
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    for (T value : values) {
      pipelineNetwork.add(value.getLayer()).freeRef();
      layers.put(value, (PipelineNetwork) pipelineNetwork.copyPipeline().freeze());
    }
    pipelineNetwork.freeRef();
  }

  @Nonnull
  public RefLinkedHashMap<T, PipelineNetwork> getLayers() {
    return new RefLinkedHashMap<>(layers);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  VisionPipeline[] addRefs(@Nullable VisionPipeline[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(VisionPipeline::addRef)
        .toArray((x) -> new VisionPipeline[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  VisionPipeline[][] addRefs(@Nullable VisionPipeline[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(VisionPipeline::addRefs)
        .toArray((x) -> new VisionPipeline[x][]);
  }

  public void _free() {
    layers.values().stream().forEach(ReferenceCountingBase::freeRef);
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  VisionPipeline<T> addRef() {
    return (VisionPipeline<T>) super.addRef();
  }
}
