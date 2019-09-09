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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.ZipSerializable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class FastPhotoStyleTransfer implements Function<Tensor, UnaryOperator<Tensor>> {

  public final Layer encode_1;
  public final Layer decode_1;
  public final Layer encode_2;
  public final Layer decode_2;
  public final Layer encode_3;
  public final Layer decode_3;
  public final Layer encode_4;
  public final Layer decode_4;
  private boolean smooth = true;
  private double lambda = 1e-4;
  private double epsilon = 1e-7;

  public FastPhotoStyleTransfer(Layer decode_1, Layer encode_1, @NotNull Layer decode_2, Layer encode_2, @NotNull Layer decode_3, Layer encode_3, @NotNull Layer decode_4, Layer encode_4) {
    this.encode_4 = encode_4;
    this.decode_4 = decode_4;
    this.encode_3 = encode_3;
    this.decode_3 = decode_3;
    this.encode_2 = encode_2;
    this.decode_2 = decode_2;
    this.encode_1 = encode_1;
    this.decode_1 = decode_1;
  }

  @Nonnull
  public static FastPhotoStyleTransfer fromZip(@Nonnull final ZipFile zipfile) {
    @Nonnull HashMap<CharSequence, byte[]> resources = ZipSerializable.extract(zipfile);
    return new FastPhotoStyleTransfer(
        Layer.fromJson(ZipSerializable.toJson(resources.get("decode_1.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("encode_1.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("decode_2.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("encode_2.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("decode_3.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("encode_3.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("decode_4.json")), resources),
        Layer.fromJson(ZipSerializable.toJson(resources.get("encode_4.json")), resources)
    );
  }

  public static Tensor transfer(Tensor contentImage, Tensor styleImage, Layer encode, Layer decode) {
    final Tensor encodedContent = encode.eval(contentImage).getDataAndFree().getAndFree(0);
    final Tensor encodedStyle = encode.eval(styleImage).getDataAndFree().getAndFree(0);
    final Tensor encodedTransformed = WCTUtil.applicator(encodedStyle).eval(encodedContent).getDataAndFree().getAndFree(0);
    return decode.eval(encodedTransformed, contentImage).getDataAndFree().getAndFree(0);
  }

  public void writeZip(@Nonnull File out, SerialPrecision precision) {
    try (@Nonnull ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(out))) {
      final HashMap<CharSequence, byte[]> resources = new HashMap<>();
      decode_1.writeZip(zipOutputStream, precision, resources, "decode_1.json");
      encode_1.writeZip(zipOutputStream, precision, resources, "encode_1.json");
      decode_2.writeZip(zipOutputStream, precision, resources, "decode_2.json");
      encode_2.writeZip(zipOutputStream, precision, resources, "encode_2.json");
      decode_3.writeZip(zipOutputStream, precision, resources, "decode_3.json");
      encode_3.writeZip(zipOutputStream, precision, resources, "encode_3.json");
      decode_4.writeZip(zipOutputStream, precision, resources, "decode_4.json");
      encode_4.writeZip(zipOutputStream, precision, resources, "encode_4.json");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Implemented process detailed in:
   * A Closed-form Solution to Photorealistic Image Stylization
   * https://arxiv.org/pdf/1802.06474.pdf
   */
  public UnaryOperator<Tensor> apply(Tensor contentImage) {
    final UnaryOperator<Tensor> photoSmooth = photoSmooth(contentImage);
    return (Tensor styleImage) -> photoSmooth.apply(photoWCT(styleImage, contentImage));
  }

  public Tensor photoWCT(Tensor style, Tensor content) {
    return photoWCT_1(style,
        photoWCT_2(style,
            photoWCT_3(style,
                photoWCT_4(style,
                    content
                ))));
  }

  public @NotNull Tensor photoWCT_1(Tensor style, Tensor content) {
    final Tensor encodedContent = encode_1.eval(content).getDataAndFree().getAndFree(0);
    final Tensor encodedStyle = encode_1.eval(style).getDataAndFree().getAndFree(0);
    final Tensor encodedTransformed = WCTUtil.applicator(encodedStyle).eval(encodedContent).getDataAndFree().getAndFree(0);
    return decode_1.eval(encodedTransformed).getDataAndFree().getAndFree(0);
  }

  @NotNull
  public Tensor photoWCT_2(Tensor style, Tensor content) {
    return transfer(content, style, encode_2, decode_2);
  }

  @NotNull
  public Tensor photoWCT_3(Tensor style, Tensor content) {
    return transfer(content, style, encode_3, decode_3);
  }

  public Tensor photoWCT_4(Tensor style, Tensor content) {
    return transfer(content, style, encode_4, decode_4);
  }

  @Nullable
  public UnaryOperator<Tensor> photoSmooth(Tensor content) {
    if (isSmooth()) return new PixelGraph(content).smoothingTransform(getLambda(), getEpsilon());
    else return x -> x;
  }

  public boolean isSmooth() {
    return smooth;
  }

  public FastPhotoStyleTransfer setSmooth(boolean smooth) {
    this.smooth = smooth;
    return this;
  }

  public double getLambda() {
    return lambda;
  }

  public FastPhotoStyleTransfer setLambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public FastPhotoStyleTransfer setEpsilon(double epsilon) {
    this.epsilon = epsilon;
    return this;
  }
}
