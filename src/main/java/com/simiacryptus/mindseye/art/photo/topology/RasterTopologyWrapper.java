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

import com.simiacryptus.mindseye.lang.Singleton;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;

public class RasterTopologyWrapper extends ReferenceCountingBase implements RasterTopology {

  public final RasterTopology inner;

  public RasterTopologyWrapper(RasterTopology inner) {
    this.inner = inner;
  }

  @Override
  public int[] getDimensions() {
    return inner.getDimensions();
  }

  @Override
  public RefList<int[]> connectivity() {
    return inner.connectivity();
  }

  @Override
  public int getIndexFromCoords(int x, int y) {
    return inner.getIndexFromCoords(x, y);
  }

  @Override
  public int[] getCoordsFromIndex(int i) {
    return inner.getCoordsFromIndex(i);
  }


  @Override
  public RasterTopologyWrapper addRef() {
    return (RasterTopologyWrapper) super.addRef();
  }

  public static class CachedRasterTopology extends RasterTopologyWrapper {

    private final Singleton<RefList<int[]>> cache = new Singleton<>();

    public CachedRasterTopology(RasterTopology inner) {
      super(inner);
    }

    @Nonnull
    @Override
    public RefList<int[]> connectivity() {
      return cache.getOrInit(() -> super.connectivity());
    }

    @Override
    protected void _free() {
      cache.freeRef();
      super._free();
    }
  }
}
