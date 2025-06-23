import 'package:flutter/material.dart';

void main() {
  runApp(const RobotApp());
}

class RobotApp extends StatelessWidget {
  const RobotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Control del Robot',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const RobotControlPage(),
    );
  }
}

class RobotControlPage extends StatefulWidget {
  const RobotControlPage({super.key});

  @override
  State<RobotControlPage> createState() => _RobotControlPageState();
}

class _RobotControlPageState extends State<RobotControlPage> {
  double _progress = 0.0;
  double _reverseProgress = 1.0;
  Offset? _tappedPosition;
  GlobalKey _imageKey = GlobalKey();

  void _startRobot() {
    print("Robot iniciat");

    setState(() {
      _progress += 0.1;
      if (_progress > 1.0) _progress = 1.0;

      _reverseProgress -= 0.1;
      if (_reverseProgress < 0.0) _reverseProgress = 0.0;
    });
  }

  void _stopRobot() {
    print("Robot aturat");

    setState(() {
      _progress -= 0.1;
      if (_progress < 0.0) _progress = 0.0;

      _reverseProgress += 0.1;
      if (_reverseProgress > 1.0) _reverseProgress = 1.0;
    });
  }

  void _resetRobot() {
    print("Robot reiniciat");

    setState(() {
      _progress = 0.0;
      _reverseProgress = 1.0;
      _tappedPosition = null;
    });
  }

  void _handleImageTap(TapDownDetails details) {
    final RenderBox renderBox = _imageKey.currentContext!.findRenderObject() as RenderBox;
    final Offset localPosition = renderBox.globalToLocal(details.globalPosition);
    
    setState(() {
      _tappedPosition = localPosition;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Control del Robot'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Sección de la imagen y coordenadas
              Column(
                children: [
                  const Text(
                    'Área de Trabajo del Robot',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.blue, width: 2),
                    ),
                    child: GestureDetector(
                      onTapDown: _handleImageTap,
                      child: Image.asset(
                        'assets/Img_Robot.jpeg', // Cambia por tu imagen
                        key: _imageKey,
                        width: double.infinity,
                        height: 200,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _tappedPosition != null
                        ? 'Coordenadas: X: ${_tappedPosition!.dx.toStringAsFixed(1)}, Y: ${_tappedPosition!.dy.toStringAsFixed(1)}'
                        : 'Toque la imagen para obtener coordenadas',
                    style: const TextStyle(fontSize: 16),
                  ),
                  const SizedBox(height: 20),
                  const Divider(thickness: 2),
                ],
              ),
              
              // Sección original de control del robot
              const SizedBox(height: 20),
              const Text('Progrés del Robot'),
              LinearProgressIndicator(
                value: _progress,
                minHeight: 20,
                backgroundColor: Colors.grey[300],
                color: Colors.green,
              ),
              const SizedBox(height: 20),
              const Text('Descarregant...'),
              LinearProgressIndicator(
                value: _reverseProgress,
                minHeight: 20,
                backgroundColor: Colors.grey[300],
                color: Colors.red,
              ),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: _startRobot,
                child: const Text('Iniciar'),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _stopRobot,
                child: const Text('Aturar'),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _resetRobot,
                child: const Text('Reset'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}