import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const RetailAssistantApp());
}

class RetailAssistantApp extends StatelessWidget {
  const RetailAssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Retail Assistant',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final List<Map<String, String>> _messages = [];
  bool _isLoading = false;
  final ScrollController _scrollController = ScrollController();

  final String apiUrl = 'ReplaceWithIP/ask'; //Replace with user IP

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _sendMessage(String text) async {
    if (text.isEmpty) return;

    setState(() {
      _messages.add({'text': text, 'isUser': 'true'});
      _isLoading = true;
    });

    _textController.clear();

    _scrollToBottom();

    try {
      final response = await http
          .post(
            Uri.parse(apiUrl),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'query': text}),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseBody = json.decode(response.body);
        final String answer = responseBody['answer'] ?? 'No answer found.';

        setState(() {
          _messages.add({'text': answer, 'isUser': 'false'});
        });
      } else {
        setState(() {
          _messages.add({
            'text': 'Error: ${response.statusCode}',
            'isUser': 'false',
          });
        });
        _showErrorDialog(
          'Failed to connect to the assistant: Server error ${response.statusCode}.',
        );
      }
    } catch (e) {
      print('Exception during API call: $e');
      setState(() {
        _messages.add({'text': 'Error: ${e.toString()}', 'isUser': 'false'});
      });
      _showErrorDialog(
        'Failed to connect to the assistant: Client exception. Please check your network or server status.',
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder:
          (ctx) => AlertDialog(
            title: const Text('Connection Error'),
            content: Text(message),
            actions: <Widget>[
              TextButton(
                child: const Text('Okay'),
                onPressed: () {
                  Navigator.of(ctx).pop();
                },
              ),
            ],
          ),
    );
  }

  Widget _buildMessage(String text, bool isUser) {
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4.0, horizontal: 8.0),
        padding: const EdgeInsets.symmetric(vertical: 10.0, horizontal: 15.0),
        decoration: BoxDecoration(
          color: isUser ? Colors.blueAccent[100] : Colors.grey[300],
          borderRadius: BorderRadius.circular(20.0),
        ),
        child: Text(text),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Retail Assistant'),
        backgroundColor: Colors.blueAccent,
        foregroundColor: Colors.white,
      ),
      resizeToAvoidBottomInset: true, // Crucial for keyboard handling
      body: Column(
        children: <Widget>[
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              reverse: false, // Display newest messages at the bottom
              padding: const EdgeInsets.all(8.0),
              itemCount: _messages.length + (_isLoading ? 1 : 0),
              itemBuilder: (context, index) {
                if (_isLoading && index == _messages.length) {
                  return const Padding(
                    padding: EdgeInsets.all(8.0),
                    child: Center(child: CircularProgressIndicator()),
                  );
                }
                final message = _messages[index];
                return _buildMessage(
                  message['text']!,
                  message['isUser'] == 'true',
                );
              },
            ),
          ),
          Padding(
            // --- ADJUSTED BOTTOM PADDING HERE ---
            // Adding a small constant (e.g., 12.0) to ensure it clears the navigation bar
            padding: EdgeInsets.fromLTRB(
              8.0, // Left
              4.0, // Top
              8.0, // Right
              MediaQuery.of(context).viewInsets.bottom +
                  12.0, // <-- INCREASED BOTTOM PADDING
            ),
            child: Row(
              children: <Widget>[
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: InputDecoration(
                      hintText: 'Ask a question...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(25.0),
                      ),
                      filled: true,
                      fillColor: Colors.white,
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 20.0,
                        vertical: 6.0,
                      ),
                    ),
                    onSubmitted: _sendMessage,
                    textInputAction: TextInputAction.send,
                  ),
                ),
                const SizedBox(width: 8.0),
                FloatingActionButton(
                  onPressed: () => _sendMessage(_textController.text),
                  backgroundColor: Colors.blueAccent,
                  foregroundColor: Colors.white,
                  mini: true,
                  child: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
