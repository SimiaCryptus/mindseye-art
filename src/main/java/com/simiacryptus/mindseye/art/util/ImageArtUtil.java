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

package com.simiacryptus.mindseye.art.util;

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.simiacryptus.hadoop_jgit.GitFileSystem;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.OptionalUploadImageQuery;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.Normalizer;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image art util.
 */
public class ImageArtUtil {

  /**
   * Gets hadoop config.
   *
   * @return the hadoop config
   */
  @Nonnull
  public static Configuration getHadoopConfig() {
    Configuration configuration = new Configuration(false);
    File tempDir = new File("temp");
    tempDir.mkdirs();
    configuration.set("hadoop.tmp.dir", tempDir.getAbsolutePath());
    configuration.set("fs.git.impl", GitFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.aws.credentials.provider", DefaultAWSCredentialsProviderChain.class.getCanonicalName());
    return configuration;
  }

  /**
   * Cuda reports closeable.
   *
   * @param log          the log
   * @param interceptLog the intercept log
   * @return the closeable
   */
  @Nonnull
  public static Closeable cudaReports(@Nonnull NotebookOutput log, boolean interceptLog) {
    Closeable handler_info = log.getHttpd().addGET("cuda/info.txt", "text/plain", outputStream -> {
      try {
        PrintStream stream = new PrintStream(outputStream);
        CudaSystem.printHeader(stream);
        stream.flush();
      } catch (Throwable e) {
        try {
          outputStream.write(Util.toString(e).getBytes("UTF-8"));
        } catch (IOException e1) {
          e1.printStackTrace();
        }
      }
    });
    Closeable handler_stats = log.getHttpd().addGET("cuda/stats.json", "application/json", outputStream -> {
      try {
        PrintStream stream = new PrintStream(outputStream);
        stream.println(JsonUtil.toJson(CudaSystem.getExecutionStatistics()));
        stream.flush();
      } catch (Throwable e) {
        try {
          outputStream.write(Util.toString(e).getBytes("UTF-8"));
        } catch (IOException e1) {
          e1.printStackTrace();
        }
      }
    });
    if (interceptLog)
      log.subreport("Cuda Logs",
          sublog -> {
            CudaSystem.addLog(new RefConsumer<String>() {
              @Nullable
              PrintWriter out;
              long remainingOut = 0;
              long killAt = 0;

              @Override
              public void accept(@Nonnull String formattedMessage) {
                if (null == out) {
                  SimpleDateFormat dateFormat = new SimpleDateFormat("dd_HH_mm_ss");
                  String date = dateFormat.format(new Date());
                  try {
                    String caption = RefString.format("Log at %s", date);
                    String filename = RefString.format("%s_cuda.log", date);
                    out = new PrintWriter(sublog.file(filename));
                    sublog.p("[%s](etc/%s)", caption, filename);
                    sublog.write();
                  } catch (Throwable e) {
                    throw Util.throwException(e);
                  }
                  killAt = RefSystem.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1);
                  remainingOut = 10L * 1024 * 1024;
                }
                out.println(formattedMessage);
                out.flush();
                int length = formattedMessage.length();
                remainingOut -= length;
                if (remainingOut < 0 || killAt < RefSystem.currentTimeMillis()) {
                  out.close();
                  out = null;
                }
              }
            });
            return null;
          });

    return new Closeable() {
      @Override
      public void close() throws IOException {
        handler_info.close();
        handler_stats.close();
      }
    };
  }

  /**
   * Convert pipeline ref map.
   *
   * @param graphDef the graph def
   * @param nodes    the nodes
   * @return the ref map
   */
  @Nonnull
  public static RefMap<String, PipelineNetwork> convertPipeline(@Nonnull GraphDef graphDef, @Nonnull String... nodes) {
    GraphModel graphModel = new GraphModel(graphDef.toByteArray());
    RefMap<String, PipelineNetwork> graphs = new RefHashMap<>();
    TFConverter tfConverter = new TFConverter();
    TFLayer tfLayer0 = new TFLayer(
        graphModel.getChild(nodes[0]).subgraph(new HashSet<>(Arrays.asList())).toByteArray(), new RefHashMap<>(),
        nodes[0], "input");
    RefUtil.freeRef(graphs.put(nodes[0], tfConverter.convert(tfLayer0)));
    for (int i = 1; i < nodes.length; i++) {
      String currentNode = nodes[i];
      String priorNode = nodes[i - 1];
      TFLayer tfLayer1 = new TFLayer(
          graphModel.getChild(currentNode).subgraph(new HashSet<>(Arrays.asList(priorNode))).toByteArray(),
          new RefHashMap<>(), currentNode, priorNode);
      RefUtil.freeRef(graphs.put(currentNode, tfConverter.convert(tfLayer1)));
    }
    return graphs;
  }

  /**
   * Load image buffered image.
   *
   * @param log       the log
   * @param image     the image
   * @param imageSize the image size
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage loadImage(@Nonnull NotebookOutput log, @Nonnull final CharSequence image, final int imageSize) {
    return Tensor.toImage(getImageTensors(image, log, imageSize)[0]);
  }

  /**
   * Load images buffered image [ ].
   *
   * @param log       the log
   * @param image     the image
   * @param imageSize the image size
   * @return the buffered image [ ]
   */
  @Nonnull
  public static BufferedImage[] loadImages(@Nonnull NotebookOutput log, @Nonnull final CharSequence image, final int imageSize) {
    return toImages(getImageTensors(image, log, imageSize));
  }

  @NotNull
  public static BufferedImage[] toImages(@Nonnull Tensor[] tensors) {
    return RefArrays.stream(tensors).map(tensor -> Tensor.toImage(tensor)).toArray(BufferedImage[]::new);
  }

  /**
   * Load images buffered image [ ].
   *
   * @param log       the log
   * @param image     the image
   * @param imageSize the image size
   * @return the buffered image [ ]
   */
  @Nonnull
  public static BufferedImage[] loadImages(@Nonnull NotebookOutput log, @Nonnull final List<? extends CharSequence> image, final int imageSize) {
    return image.stream().flatMap(img -> Arrays.stream(getImageTensors(img, log, imageSize)))
        .map(tensor -> Tensor.toImage(tensor)).toArray(BufferedImage[]::new);
  }

  /**
   * Load image buffered image.
   *
   * @param log    the log
   * @param image  the image
   * @param width  the width
   * @param height the height
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage loadImage(@Nonnull NotebookOutput log, @Nonnull final CharSequence image, final int width, final int height) {
    return Tensor.toImage(getImageTensors(image, log, width, height)[0]);
  }

  /**
   * Load images buffered image [ ].
   *
   * @param log    the log
   * @param image  the image
   * @param width  the width
   * @param height the height
   * @return the buffered image [ ]
   */
  @Nonnull
  public static BufferedImage[] loadImages(@Nonnull NotebookOutput log, @Nonnull final CharSequence image, final int width, final int height) {
    return toImages(getImageTensors(image, log, width, height));
  }

  /**
   * Load images buffered image [ ].
   *
   * @param log    the log
   * @param image  the image
   * @param width  the width
   * @param height the height
   * @return the buffered image [ ]
   */
  @Nonnull
  public static BufferedImage[] loadImages(@Nonnull NotebookOutput log, @Nonnull final List<? extends CharSequence> image, final int width, final int height) {
    return image.stream().flatMap(img -> Arrays.stream(getImageTensors(img, log, width, height)))
        .map(tensor -> Tensor.toImage(tensor)).toArray(BufferedImage[]::new);
  }

  /**
   * Gets image tensor.
   *
   * @param file  the file
   * @param log   the log
   * @param width the width
   * @return the image tensor
   */
  @Nonnull
  public static Tensor[] getImageTensors(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width) {
    String fileStr = file.toString();
//    int length = fileStr.split("\\:")[0].length();
//    if (length <= 0 || length >= Math.min(7, fileStr.length())) {
//    }
    if(fileStr.contains(",")) {
      return Arrays.stream(fileStr.split(",")).flatMap(x-> Arrays.stream(getImageTensors(x, log, width))).toArray(Tensor[]::new);
    } else if (fileStr.contains(" + ")) {
      String[] split = fileStr.split(" +\\+ +");
      Tensor[] sampleImages = getImageTensors(split[0], log, width);
      return Arrays.stream(sampleImages).map(sampleImage->{
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(split)
            .flatMap(x -> Arrays.stream(getImageTensors(x, log, sampleImageDimensions[0], sampleImageDimensions[1])))
            .reduce((a, b) -> {
              Tensor r = a.mapCoords(c -> a.get(c) + b.get(c));
              b.freeRef();
              a.freeRef();
              return r;
            }));
      }).toArray(Tensor[]::new);
    } else if (fileStr.contains(" * ")) {
      String[] split = fileStr.split(" +\\* +");
      Tensor sampleImage = RefUtil.get(RefArrays.stream(split).flatMap(x -> Arrays.stream(getImageTensors(x, log, width))).findFirst());
      int[] sampleImageDimensions = sampleImage.getDimensions();
      sampleImage.freeRef();
      return new Tensor[]{
        RefUtil.get(RefArrays.stream(split)
                .flatMap(x -> Arrays.stream(getImageTensors(x, log, sampleImageDimensions[0], sampleImageDimensions[1])))
                .reduce((a, b) -> {
                  Tensor r = a.mapCoords(c -> a.get(c) * b.get(c));
                  b.freeRef();
                  a.freeRef();
                  return r;
                }))
      };
    } else if (fileStr.trim().toLowerCase().equals("plasma")) {
      return new Tensor[]{ new Plasma().paint(width, width) };
    } else if (fileStr.trim().toLowerCase().equals("noise")) {
      Tensor baseTensor = new Tensor(width, width, 3);
      Tensor map = baseTensor.map(x -> FastRandom.INSTANCE.random() * 100);
      baseTensor.freeRef();
      return new Tensor[]{map};
    } else if (fileStr.trim().toLowerCase().startsWith("upload:")) {
      String key = fileStr.substring("upload:".length());
      return (Tensor[]) MarkdownNotebookOutput.uploadCache.computeIfAbsent(key, (RefFunction<String, Tensor[]>) k -> {
        try {
          RefArrayList<Tensor> tensors = new RefArrayList<>();
          while (true) {
            OptionalUploadImageQuery uploadImageQuery = new OptionalUploadImageQuery(k, log);
            Optional<File> optionalFile = uploadImageQuery.print().get();
            if (optionalFile.isPresent()) {
              tensors.add(Tensor.fromRGB(ImageIO.read(optionalFile.get())));
            } else {
              break;
            }
          }
          Tensor[] array = tensors.toArray(new Tensor[]{});
          tensors.freeRef();
          return array;
        } catch (IOException e) {
          throw Util.throwException(e);
        }
      });
    } else if (fileStr.trim().toLowerCase().startsWith("artist:")) {
      String[] paintings = getPaintingsByArtist(stripPrefix("artist:", fileStr.trim().toLowerCase()).split(" ")[0], width);
      return Arrays.stream(paintings).map(f->{
        Tensor tensor = loadImageFile(f);
        Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, true));
        tensor.freeRef();
        return resized;
      }).toArray(Tensor[]::new);
    } else if (fileStr.trim().toLowerCase().startsWith("wikiart:")) {
      String[] paintings = getPaintingsBySearch(stripPrefix("wikiart:", fileStr.trim().toLowerCase()).split(" ")[0].replace("&"," "), width);
      return Arrays.stream(paintings).map(f->{
        Tensor tensor = loadImageFile(f);
        Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, true));
        tensor.freeRef();
        return resized;
      }).toArray(Tensor[]::new);
    } else if (isDouble(fileStr)) {
      double v = Double.parseDouble(fileStr);
      Tensor baseTensor = new Tensor(width, width, 3);
      Tensor map = baseTensor.map(x -> v);
      baseTensor.freeRef();
      return new Tensor[]{map};
    } else {
      Tensor tensor = loadImageFile(file);
      Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, true));
      tensor.freeRef();
      return new Tensor[]{resized};
    }
  }

  public static String[] getPaintingsBySearch(String searchWord, int minWidth) {
    return getPaintings(urlPaintingsBySearch(searchWord), 100, minWidth(minWidth));
  }

  @NotNull
  public static URI urlPaintingsBySearch(String searchWord) {
    try {
      return new URI("https://www.wikiart.org/en/search/" + URLEncoder.encode(searchWord, "UTF-8").replaceAll("\\+", "%20") + "/1?json=2");
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException(e);
    }
  }

  public static String[] getPaintingsByArtist(String artist, int minWidth) {
    try {
      return getPaintings(urlPaintingsByArtist(artist), 100, minWidth(minWidth));
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    }
  }

  @NotNull
  public static URI urlPaintingsByArtist(String artist) throws URISyntaxException {
    return new URI("https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl=" + artist);
  }

  @NotNull
  public static Predicate<Map<String, Object>> minWidth(int minWidth) {
    return x -> ((Number) x.get("width")).doubleValue() > minWidth;
  }

  @NotNull
  public static String[] getPaintings(URI uri, int maxResults, Predicate<Map<String, Object>>... filterFns) {
    try {
      File cacheDir = new File("wikiart");
      Type type = new TypeToken<ArrayList<Map<String, Object>>>() {}.getType();
      String json = IOUtils.toString(uri, "UTF-8");
      ArrayList<Map<String, Object>> result = new GsonBuilder().create().fromJson(json, type);

      Stream<Map<String, Object>> stream = result.stream();
      for(Predicate<Map<String, Object>> filterFn : filterFns) {
        stream = stream.filter(filterFn);
      }
      return stream
              .map(x -> x.get("image").toString())
              .map(x -> stripSuffix("!Large.jpg", x))
              .limit(maxResults)
              .map(file -> {
                try {
                  String fileName = Normalizer.normalize(
                          Arrays.stream(takeRight(2, file.split("/"))).reduce((a, b) -> a + "/" + b).get(),
                          Normalizer.Form.NFD
                  ).replaceAll("[^\\p{ASCII}]", "");
                  File localFile = new File(cacheDir, fileName);
                  if (!localFile.exists()) {
                    FileUtils.writeByteArrayToFile(localFile, IOUtils.toByteArray(new URI(encodeURL(file))));
                  }
                  return "file:///" + stripPrefix("/", localFile.getAbsolutePath().replaceAll("\\\\", "/"));
                } catch (IOException e) {
                  throw new RuntimeException(e);
                } catch (URISyntaxException e) {
                  throw new RuntimeException(e);
                }
              })
              .filter(x->x != null)
              .toArray(String[]::new);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @NotNull
  public static String encodeURL(String file) {
    return Arrays.stream(file.split("/")).map(s -> {
      try {
        return s.endsWith(":") ? s : URLEncoder.encode(s, "UTF-8");
      } catch (UnsupportedEncodingException e) {
        throw new RuntimeException(e);
      }
    }).reduce((a, b) -> a + "/" + b).get();
  }

  public static String[] takeRight(int n, String[] split) {
    return Arrays.copyOfRange(split, Math.max(0, split.length-n), split.length);
  }

  public static String stripPrefix(String prefix, String str) {
    if (str.startsWith(prefix)) {
      return str.substring(prefix.length());
    } else {
      return str;
    }
  }

  public static String stripSuffix(String prefix, String str) {
    if (str.endsWith(prefix)) {
      return str.substring(0, str.length() - prefix.length());
    } else {
      return str;
    }
  }

  /**
   * Gets image tensor.
   *
   * @param file   the file
   * @param log    the log
   * @param width  the width
   * @param height the height
   * @return the image tensor
   */
  @Nonnull
  public static Tensor[] getImageTensors(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width, int height) {
    String fileStr = file.toString().trim();
//    int length = fileStr.split("\\:")[0].length();
//    if (length <= 0 || length >= Math.min(7, fileStr.length())) {
//    }
    if(fileStr.contains(",")) {
      return Arrays.stream(fileStr.split(",")).flatMap(x-> Arrays.stream(getImageTensors(x, log, width, height))).toArray(Tensor[]::new);
    } else if (fileStr.contains(" + ")) {
      return Arrays.stream(getImageTensors(fileStr.split(" +\\+ +")[0], log, width, height)).map(sampleImage->{
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\+ +"))
            .flatMap(x -> Arrays.stream(getImageTensors(x, log, sampleImageDimensions[0], sampleImageDimensions[1])))
            .reduce((a, b) -> {
              Tensor r = a.mapCoords(c -> Math.min(255, Math.max(0, a.get(c) + b.get(c))));
              a.freeRef();
              b.freeRef();
              return r;
            }));
      }).toArray(Tensor[]::new);
    } else if (fileStr.contains(" * ")) {
      return Arrays.stream(getImageTensors(fileStr.split(" +\\* +")[0], log, width, height)).map(sampleImage->{
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\* +"))
            .flatMap(x -> Arrays.stream(getImageTensors(x, log, sampleImageDimensions[0], sampleImageDimensions[1])))
            .reduce((a, b) -> {
              try {
                return a.mapCoords(c -> Math.min(255, Math.max(0, a.get(c) * b.get(c))));
              } finally {
                b.freeRef();
                a.freeRef();
              }
            }));
      }).toArray(Tensor[]::new);
      //Tensor sampleImage =
    } else if (fileStr.trim().toLowerCase().startsWith("upload:")) {
      String key = fileStr.substring("upload:".length());
      return (Tensor[]) MarkdownNotebookOutput.uploadCache.computeIfAbsent(key, (RefFunction<String, Tensor[]>) k -> {
        try {
          RefArrayList<Tensor> tensors = new RefArrayList<>();
          while (true) {
            OptionalUploadImageQuery uploadImageQuery = new OptionalUploadImageQuery(k, log);
            Optional<File> optionalFile = uploadImageQuery.print().get();
            if (optionalFile.isPresent()) {
              tensors.add(Tensor.fromRGB(ImageIO.read(optionalFile.get())));
            } else {
              break;
            }
          }
          Tensor[] array = tensors.toArray(new Tensor[]{});
          tensors.freeRef();
          return array;
        } catch (IOException e) {
          throw Util.throwException(e);
        }
      });
    } else if (fileStr.trim().toLowerCase().startsWith("wikiart:")) {
      String[] paintings = getPaintingsBySearch(stripPrefix("wikiart:", fileStr.trim().toLowerCase()).split(" ")[0].replace("&"," "), width);
      return Arrays.stream(paintings).map(f->{
        Tensor tensor = loadImageFile(f);
        Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, height));
        tensor.freeRef();
        return resized;
      }).toArray(Tensor[]::new);
    } else if (fileStr.trim().toLowerCase().startsWith("artist:")) {
      String[] paintings = getPaintingsByArtist(stripPrefix("artist:", fileStr.trim().toLowerCase()).split(" ")[0], width);
      return Arrays.stream(paintings).map(f->{
        Tensor tensor = loadImageFile(f);
        Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, height));
        tensor.freeRef();
        return resized;
      }).toArray(Tensor[]::new);
    } else if (fileStr.trim().toLowerCase().equals("plasma")) {
      return new Tensor[]{new Plasma().paint(width, height)};
    } else if (fileStr.trim().toLowerCase().equals("noise")) {
      Tensor dims = new Tensor(width, height, 3);
      try {
        return new Tensor[]{dims.map(x -> FastRandom.INSTANCE.random() * 100)};
      } finally {
        dims.freeRef();
      }
    } else if (isDouble(fileStr)) {
      Tensor dims = new Tensor(width, height, 3);
      try {
        double v = Double.parseDouble(fileStr);
        return new Tensor[]{dims.map(x -> v)};
      } finally {
        dims.freeRef();
      }
    } else {
      Tensor tensor = loadImageFile(file);
      Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, height));
      tensor.freeRef();
      return new Tensor[]{resized};
    }
  }

  public static boolean isDouble(String str) {
    return str.matches("\\-?\\d+(?:\\.\\d*)?(?:[eE]\\-?\\d+)?");
  }

  /**
   * Gets tensor.
   *
   * @param file the file
   * @return the tensor
   */
  @Nonnull
  public static Tensor loadImageFile(@Nonnull CharSequence file) {
    String fileStr = file.toString();
    if (fileStr.startsWith("http")) {
      try {
        File cacheFile = new File("http_cache", new Path(fileStr).getName());
        if(!cacheFile.exists()) {
          cacheFile.getParentFile().mkdirs();
          try(InputStream inputStream = new URL(fileStr).openStream()) {
            IOUtils.copy(inputStream, new FileOutputStream(cacheFile));
          }
        }
        BufferedImage read = ImageIO.read(cacheFile);
        if (null == read)
          throw new IllegalArgumentException("Error reading " + file);
        return Tensor.fromRGB(read);
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    } else {
      FileSystem fileSystem = getFileSystem(fileStr);
      Path path = new Path(fileStr);
      try {
        if (!fileSystem.exists(path))
          throw new IllegalArgumentException("Not Found: " + path);
        try (FSDataInputStream open = fileSystem.open(path)) {
          byte[] bytes = IOUtils.toByteArray(open);
          try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) {
            return Tensor.fromRGB(ImageIO.read(in));
          }
        }
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    }
  }

  /**
   * Gets file system.
   *
   * @param file the file
   * @return the file system
   */
  public static FileSystem getFileSystem(@Nonnull final CharSequence file) {
    Configuration conf = getHadoopConfig();
    FileSystem fileSystem;
    try {
      fileSystem = FileSystem.get(new Path(file.toString()).toUri(), conf);
    } catch (IOException e) {
      throw Util.throwException(e);
    }
    return fileSystem;
  }

  /**
   * Get index map int [ ] [ ].
   *
   * @param layer the layer
   * @return the int [ ] [ ]
   */
  @Nonnull
  public static int[][] getIndexMap(@Nonnull final SimpleConvolutionLayer layer) {
    int[] kernelDimensions = layer.getKernelDimensions();
    layer.freeRef();
    double b = Math.sqrt(kernelDimensions[2]);
    int h = kernelDimensions[1];
    int w = kernelDimensions[0];
    int l = (int) (w * h * b);
    return IntStream.range(0, (int) b).mapToObj(i -> {
      return IntStream.range(0, l).map(j -> j + l * i).toArray();
    }).toArray(i -> new int[i][]);
  }


}
