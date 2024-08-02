import 'dart:io';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'dart:math';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  dynamic _probability = 0;
  String? _result;
  List<String>? _labels;
  late tfl.Interpreter _interpreter;
  final picker = ImagePicker();

  String _selected0 = "";
  String _selected1 = "";
  String val0 = "";
  String val1 = "";
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      loadLabels().then((loadedLabels) {
        setState(() {
          _labels = loadedLabels;
        });
      });
    });
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Container(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: <Widget>[
                const SizedBox(height: 80),
                const Text(
                  'Image Quality Detector',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      height: 1.4,
                      fontFamily: 'SofiaSans',
                      fontSize: 30),
                ),
                const SizedBox(height: 50),
                Center(
                    child: SizedBox(
                  width: 350,
                  child: Column(
                    children: <Widget>[
                      _imagePreview(_image),
                      const SizedBox(height: 50),
                    ],
                  ),
                )),
                SizedBox(
                  width: MediaQuery.of(context).size.width,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: <Widget>[
                      Expanded(
                        child: GestureDetector(
                          onTap: () {
                            pickImageFromCamera();
                          },
                          child: Container(
                            alignment: Alignment.center,
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 18),
                            decoration: BoxDecoration(
                              color: Colors.black38,
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Text(
                              'Capture a Photo',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  fontFamily: 'SofiaSans'),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 20),
                      Expanded(
                        child: GestureDetector(
                          onTap: () {
                            pickImageFromGallery();
                          },
                          child: Container(
                            alignment: Alignment.center,
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 18),
                            decoration: BoxDecoration(
                              color: Colors.black38,
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Text(
                              'Select a photo',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  fontFamily: 'SofiaSans'),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            )));
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/model.tflite');
    } catch (e) {
      debugPrint('Error loading model: $e');
    }
  }

  Future<void> pickImageFromCamera() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      _setImage(File(pickedFile.path));
    }
  }

  Future<void> pickImageFromGallery() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      _setImage(File(pickedFile.path));
    }
  }

  void _setImage(File image) async {
    setState(() {
      _isLoading = true;
    });

    _image = image;
    await runInference(image);

    setState(() {
      _isLoading = false;
    });
  }

  Future<Uint8List> preprocessImage(File imageFile) async {
    try {
      // Decode the image to an Image object
      img.Image? originalImage = img.decodeImage(await imageFile.readAsBytes());

      if (originalImage == null) {
        throw Exception("Unable to decode image.");
      }

      // Resize the image to the correct size (224x224)
      img.Image resizedImage =
          img.copyResize(originalImage, width: 224, height: 224);

      // Convert the image to RGB bytes and remove the alpha channel if present
      List<int> rgbBytes = [];
      for (int y = 0; y < resizedImage.height; y++) {
        for (int x = 0; x < resizedImage.width; x++) {
          int pixel = resizedImage.getPixel(x, y);
          rgbBytes.add(img.getRed(pixel));
          rgbBytes.add(img.getGreen(pixel));
          rgbBytes.add(img.getBlue(pixel));
        }
      }

      // Ensure the output is a Uint8List
      return Uint8List.fromList(rgbBytes);
    } catch (e) {
      throw Exception(e);
    }
  }

  Float32List normalizeInput(Uint8List inputBytes) {
    try {
      // Normalize the input bytes to the range [0, 1] if required by the model
      Float32List normalizedInput = Float32List(inputBytes.length);
      for (int i = 0; i < inputBytes.length; i++) {
        normalizedInput[i] = inputBytes[i] / 255.0;
      }
      return normalizedInput;
    } catch (e) {
      throw Exception(e);
    }
  }

  Future<void> runInference(
    File? _image,
  ) async {
    if (_labels == null || _image == null) {
      throw Exception(
          "Error on runInference : _labels or _image are not Available ");
    }

    try {
      Uint8List inputBytes = await preprocessImage(_image);
      Float32List input = normalizeInput(inputBytes);

      // Check if the input tensor matches the expected shape and data type
      var inputTensor = input.buffer.asFloat32List().reshape([1, 224, 224, 3]);
      var outputBuffer = List<double>.filled(1 * 2, 0).reshape([1, 2]);

      _interpreter.run(inputTensor, outputBuffer);

      // Process the output
      List<double> output = outputBuffer[0];
      double maxScore = output.reduce(max);
      dynamic _probability = maxScore;

      int highestProbIndex = output.indexOf(maxScore);
      String classificationResult = _labels![highestProbIndex];
      String? _result = classificationResult.trim();
      // double _pro = (_probability * 100);

      setState(() {
        if (_result == "BLUR") {
          _selected0 = "BLUR";
          val0 = '${(_probability * 100).toStringAsFixed(0)}%';
        } else {
          _selected0 = '';
          val0 = '${(100 - (_probability * 100)).toStringAsFixed(0)}%';
        }

        if (_result.toString() == "SHARP") {
          _selected1 = "SHARP";
          val1 = '${(_probability * 100).toStringAsFixed(0)}%';
        } else {
          _selected1 = "";
          val1 = '${(100 - (_probability * 100)).toStringAsFixed(0)}%';
        }
      });

      // navigateToResult();
    } catch (e) {
      debugPrint('Error during inference: $e');
      throw Exception(e);
    }
  }

  Future<List<String>> loadLabels() async {
    final labelsData =
        await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    return labelsData.split('\n');
  }

  String classifyImage(List<int> output) {
    int highestProbIndex = output.indexOf(output.reduce(max));
    return _labels![highestProbIndex];
  }

  _imagePreview(File? image) {
    return Stack(
      alignment: AlignmentDirectional.center,
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Container(
                margin: const EdgeInsets.all(15.0),
                padding: const EdgeInsets.all(3.0),
                decoration:
                    BoxDecoration(border: Border.all(color: Colors.blueAccent)),
                height: 300,
                child: image != null
                    ? Image.file(image)
                    : const Center(
                        child: Text('Image'),
                      )),
            const SizedBox(
              height: 5,
            ),
            Card(
                child: RadioListTile<String>(
                    activeColor: Theme.of(context).primaryColor,
                    groupValue: _selected0,
                    value: "BLUR",
                    onChanged: (String? value) {},
                    title: const Text("BLUR", style: TextStyle(fontSize: 16.0)),
                    subtitle: Text(val0))),
            Card(
                child: RadioListTile<String>(
                    activeColor: Theme.of(context).primaryColor,
                    groupValue: _selected1,
                    value: "SHARP",
                    onChanged: (String? value) {},
                    title:
                        const Text("SHARP", style: TextStyle(fontSize: 16.0)),
                    subtitle: Text(val1))),
          ],
        ),
        if (_isLoading) const Text('PLease wait....')
      ],
    );
  }
}
