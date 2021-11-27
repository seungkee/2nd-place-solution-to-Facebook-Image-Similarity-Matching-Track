from models import encoder
from models import losses
from models import resnet
from models import ssl
from models import vision_transformer
REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
}
