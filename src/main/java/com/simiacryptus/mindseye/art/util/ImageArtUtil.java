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
import com.simiacryptus.notebook.UploadImageQuery;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

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
    Tensor imageTensor = getImageTensor(image, log, imageSize);
    BufferedImage bufferedImage = imageTensor.toImage();
    imageTensor.freeRef();
    return bufferedImage;
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
    return RefArrays.stream(getImageTensors(image, log, imageSize)).map(tensor -> {
      BufferedImage bufferedImage = tensor.toImage();
      tensor.freeRef();
      return bufferedImage;
    }).toArray(BufferedImage[]::new);
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
        .map(tensor -> {
          BufferedImage bufferedImage = tensor.toImage();
          tensor.freeRef();
          return bufferedImage;
        }).toArray(BufferedImage[]::new);
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
    Tensor imageTensor = getImageTensor(image, log, width, height);
    BufferedImage bufferedImage = imageTensor.toImage();
    imageTensor.freeRef();
    return bufferedImage;
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
    return RefArrays.stream(getImageTensors(image, log, width, height))
        .map(tensor -> {
          BufferedImage img = tensor.toImage();
          tensor.freeRef();
          return img;
        }).toArray(BufferedImage[]::new);
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
        .map(tensor -> {
          BufferedImage img = tensor.toImage();
          tensor.freeRef();
          return img;
        }).toArray(BufferedImage[]::new);
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
  public static Tensor getImageTensor(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width) {
    String fileStr = file.toString();
    int length = fileStr.split("\\:")[0].length();
    if (length <= 0 || length >= Math.min(7, fileStr.length())) {
      if (fileStr.contains(" + ")) {
        Tensor sampleImage = getImageTensor(fileStr.split(" +\\+ +")[0], log, width);
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\+ +"))
            .map(x -> getImageTensor(x, log, sampleImageDimensions[0], sampleImageDimensions[1]))
            .reduce((a, b) -> {
              Tensor r = a.mapCoords(c -> a.get(c) + b.get(c));
              b.freeRef();
              a.freeRef();
              return r;
            }));
      } else if (fileStr.contains(" * ")) {
        Tensor sampleImage = RefUtil.get(RefArrays.stream(fileStr.split(" +\\* +")).map(x -> getImageTensor(x, log, width)).findFirst());
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\* +"))
            .map(x -> getImageTensor(x, log, sampleImageDimensions[0], sampleImageDimensions[1]))
            .reduce((a, b) -> {
              Tensor r = a.mapCoords(c -> a.get(c) * b.get(c));
              b.freeRef();
              a.freeRef();
              return r;
            }));
      } else if (fileStr.trim().toLowerCase().equals("plasma")) {
        return new Plasma().paint(width, width);
      } else if (fileStr.trim().toLowerCase().equals("noise")) {
        Tensor baseTensor = new Tensor(width, width, 3);
        Tensor map = baseTensor.map(x -> FastRandom.INSTANCE.random() * 100);
        baseTensor.freeRef();
        return map;
      } else if (fileStr.matches("\\-?\\d+(?:\\.\\d*)?(?:[eE]\\-?\\d+)?")) {
        double v = Double.parseDouble(fileStr);
        Tensor baseTensor = new Tensor(width, width, 3);
        Tensor map = baseTensor.map(x -> v);
        baseTensor.freeRef();
        return map;
      }
    }
    Tensor tensor = getTensor(log, file);
    Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, true));
    tensor.freeRef();
    return resized;
  }

  /**
   * Get image tensors tensor [ ].
   *
   * @param file  the file
   * @param log   the log
   * @param width the width
   * @return the tensor [ ]
   */
  @Nonnull
  public static Tensor[] getImageTensors(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width) {
    return RefArrays.stream(getTensors(log, file))
        .map(tensor -> {
          BufferedImage bufferedImage = tensor.toImage();
          tensor.freeRef();
          return bufferedImage;
        }).map(image -> ImageUtil.resize(image, width, true))
        .map(resize -> Tensor.fromRGB(resize))
        .toArray(Tensor[]::new);
  }

  /**
   * Get image tensors tensor [ ].
   *
   * @param file   the file
   * @param log    the log
   * @param width  the width
   * @param height the height
   * @return the tensor [ ]
   */
  @Nonnull
  public static Tensor[] getImageTensors(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width, int height) {
    return RefArrays.stream(getTensors(log, file))
        .map(tensor -> {
          BufferedImage img = tensor.toImage();
          tensor.freeRef();
          return img;
        })
        .map(image -> ImageUtil.resize(image, width, height))
        .map(resize -> Tensor.fromRGB(resize))
        .toArray(Tensor[]::new);
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
  public static Tensor getImageTensor(@Nonnull final CharSequence file, @Nonnull NotebookOutput log, int width, int height) {
    String fileStr = file.toString();
    int length = fileStr.split("\\:")[0].length();
    if (length <= 0 || length >= Math.min(7, fileStr.length())) {
      if (fileStr.contains(" + ")) {
        Tensor sampleImage = getImageTensor(fileStr.split(" +\\+ +")[0], log, width, height);
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\+ +"))
            .map(x -> getImageTensor(x, log, sampleImageDimensions[0], sampleImageDimensions[1]))
            .reduce((a, b) -> {
              Tensor r = a.mapCoords(c -> Math.min(255, Math.max(0, a.get(c) + b.get(c))));
              a.freeRef();
              b.freeRef();
              return r;
            }));
      } else if (fileStr.contains(" * ")) {
        Tensor sampleImage = getImageTensor(fileStr.split(" +\\* +")[0], log, width, height);
        ;
        int[] sampleImageDimensions = sampleImage.getDimensions();
        sampleImage.freeRef();
        return RefUtil.get(RefArrays.stream(fileStr.split(" +\\* +"))
            .map(x -> getImageTensor(x, log, sampleImageDimensions[0], sampleImageDimensions[1]))
            .reduce((a, b) -> {
              try {
                return a.mapCoords(c -> Math.min(255, Math.max(0, a.get(c) * b.get(c))));
              } finally {
                b.freeRef();
                a.freeRef();
              }
            }));
      } else if (fileStr.trim().toLowerCase().equals("plasma")) {
        return new Plasma().paint(width, height);
      } else {
        Tensor dims = new Tensor(width, height, 3);
        try {
          if (fileStr.trim().toLowerCase().equals("noise")) {
            return dims.map(x -> FastRandom.INSTANCE.random() * 100);
          } else if (fileStr.matches("\\-?\\d+(?:\\.\\d*)?(?:[eE]\\-?\\d+)?")) {
            double v = Double.parseDouble(fileStr);
            return dims.map(x -> v);
          }
        } finally {
          dims.freeRef();
        }
      }
    }
    Tensor tensor = getTensor(log, file);
    Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), width, height));
    tensor.freeRef();
    return resized;
  }

  /**
   * Get tensors tensor [ ].
   *
   * @param log  the log
   * @param file the file
   * @return the tensor [ ]
   */
  @Nonnull
  public static Tensor[] getTensors(@Nonnull NotebookOutput log, @Nonnull CharSequence file) {
    String fileStr = file.toString();
    if(fileStr.contains(",")) {
      return Arrays.stream(fileStr.split(",")).flatMap(x-> Arrays.stream(getTensors(log, x))).toArray(Tensor[]::new);
    }
    try {
      String uploadPrefix = "upload:";
      if (fileStr.trim().toLowerCase().startsWith(uploadPrefix)) {
        String key = fileStr.substring(uploadPrefix.length());
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
      } else if (fileStr.startsWith("http")) {
        BufferedImage read = ImageIO.read(new URL(fileStr));
        if (null == read)
          throw new IllegalArgumentException("Error reading " + file);
        return new Tensor[]{Tensor.fromRGB(read)};
      } else {
        RefArrayList<Tensor> tensors = new RefArrayList<>();
        if (!fileStr.isEmpty()) {
          FileSystem fileSystem = getFileSystem(fileStr);
          Path path = new Path(fileStr);
          if (!fileSystem.exists(path)) {
            tensors.freeRef();
            throw new IllegalArgumentException("Not Found: " + path);
          }
          RemoteIterator<LocatedFileStatus> iterator = fileSystem.listFiles(path, true);
          while (iterator.hasNext()) {
            LocatedFileStatus locatedFileStatus = iterator.next();
            try (FSDataInputStream open = fileSystem.open(locatedFileStatus.getPath())) {
              byte[] bytes = IOUtils.toByteArray(open);
              try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) {
                tensors.add(Tensor.fromRGB(ImageIO.read(in)));
              }
            }
          }
        } else {
          tensors.add(new Tensor(256, 256, 3));
        }
        Tensor[] array = tensors.toArray(new Tensor[]{});
        tensors.freeRef();
        return array;
      }
    } catch (Throwable e) {
      throw new RuntimeException("Error reading " + file, e);
    }
  }

  /**
   * Gets tensor.
   *
   * @param log  the log
   * @param file the file
   * @return the tensor
   */
  @Nonnull
  public static Tensor getTensor(@Nonnull NotebookOutput log, @Nonnull CharSequence file) {
    String fileStr = file.toString();
    if (fileStr.trim().toLowerCase().startsWith("upload:")) {
      String key = fileStr.substring("upload:".length());
      return (Tensor) MarkdownNotebookOutput.uploadCache.computeIfAbsent(key, (RefFunction<String, Tensor>) k -> {
        try {
          UploadImageQuery uploadImageQuery = new UploadImageQuery(k, log);
          return Tensor.fromRGB(ImageIO.read(uploadImageQuery.print().get()));
        } catch (IOException e) {
          throw Util.throwException(e);
        }
      });
    } else if (fileStr.startsWith("http")) {
      try {
        BufferedImage read = ImageIO.read(new URL(fileStr));
        if (null == read)
          throw new IllegalArgumentException("Error reading " + file);
        return Tensor.fromRGB(read);
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    }
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
