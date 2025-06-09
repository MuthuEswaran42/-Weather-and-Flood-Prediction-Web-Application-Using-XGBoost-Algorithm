# projectname/routing.py
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from chat import consumers
from django.urls import path

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/chat/<str:room_name>/", consumers.ChatConsumer.as_asgi()),
        ])
    ),
})
