import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://127.0.0.1:42070")

# Simulate Kimodo finishing a generation and sending the path
socket.send_string(r"K:\project_kaspar\modules\kai-avatar-animation\kimodo\kimodo-gen\carry.bvh")
print("Dummy signal sent to Blender.")