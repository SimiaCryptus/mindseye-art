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

package com.simiacryptus.mindseye.art.photo.affinity;

import java.util.List;

/**
 * The type Affinity wrapper.
 */
public abstract class AffinityWrapper implements RasterAffinity {

  /**
   * The Inner.
   */
  RasterAffinity inner;

  /**
   * Instantiates a new Affinity wrapper.
   *
   * @param inner the inner
   */
  public AffinityWrapper(RasterAffinity inner) {
    this.inner = inner;
  }

  @Override
  public abstract List<double[]> affinityList(List<int[]> graphEdges);

}
