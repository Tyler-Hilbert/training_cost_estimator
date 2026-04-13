// Makes a request to Modal using data on webpage, then updates webpage.

const MODAL_URL = "https://hilberttyler1--training-cost-estimator-v2-training-cost.modal.run";

const module_id = "module"
const input_shape_id = "input_shape"
const loss_fn_id = "loss_fn"
const optimizer_id = "optimizer"
const num_examples_id = "num_examples"
const compile_option_query = 'input[name="compile_option"]:checked'
const compile_set_value = "use-torch-compile"
const target_hardware_query = 'input[name="target_hardware"]:checked'
const submit_id = "submit"
const price_per_epoch_id = "price_per_epoch"
const result_id = "result"
const error_id = "error"


async function send_request() {
    // Clear UI
    document.getElementById(submit_id).disabled = true
    document.getElementById(submit_id).textContent = "This may take a minute..."
    document.getElementById(price_per_epoch_id).textContent = ""
    document.getElementById(result_id).textContent = "";
    document.getElementById(error_id).textContent = "";

    try {
        // Make request to Modal
        const response = await fetch(MODAL_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                module_code: document.getElementById(module_id).value,
                shape_code: document.getElementById(input_shape_id).value,
                loss_function: document.getElementById(loss_fn_id).value,
                optimizer: document.getElementById(optimizer_id).value,
                training_examples: document.getElementById(num_examples_id).value,
                target_hardware: document.querySelector(target_hardware_query).value,
                compiler: document.querySelector(compile_option_query).id === compile_set_value
            })
        });
        const data = await response.json();
        
        // Update UI with results
        document.getElementById(price_per_epoch_id).textContent = data.price_per_epoch
        const resultEl = document.getElementById(result_id);
        resultEl.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

    } catch (err) {
        // Error
        // FIXME display error
        document.getElementById(error_id).textContent = "Error: contact HilbertTyler1@gmail.com for demo.";
    } finally {
        // Reset button
        document.getElementById(submit_id).disabled = false
        document.getElementById(submit_id).textContent = "Estimate Training Cost";
    }
}

const PRESETS = {
mlp: {
module_code: `
class Model(nn.Module):
    def __init__(self, input_size=3072, layer_sizes=[512, 256], output_size=10):
        super(Model, self).__init__()
        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
`,
shape_code: `
batch_size = 128
input_size = 3072
input_shape = (batch_size, input_size)
`,
loss_function: "nn.CrossEntropyLoss()",
optimizer: "torch.optim.SGD(model.parameters(), lr=1e-3)",
training_examples: "1000000"
},

alexnet: {
module_code: `
class Model(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
`,
shape_code: `
batch_size = 128
input_shape = (batch_size, 3, 224, 224)
`,
loss_function: "nn.CrossEntropyLoss()",
optimizer: "torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)",
training_examples: "1000000"
},

resnet50: {
module_code: `
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * 4, stride),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))
`,
shape_code: `
batch_size = 128
input_shape = (batch_size, 3, 224, 224)
`,
loss_function: "nn.CrossEntropyLoss()",
optimizer: "torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)",
training_examples: "1000000"
},

transformer: {
module_code: `
class Model(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(Model, self).__init__()
        # Parameters exactly from Table 1 (Base Model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.classifier = nn.Linear(d_model, 1000)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Using x as both src and tgt to simulate the workload
        out = self.transformer(x, x)
        return self.classifier(out.mean(dim=1))
`,
shape_code: `
batch_size = 48
seq_len = 512
d_model = 512
input_shape = (batch_size, seq_len, d_model)
`,
loss_function: "nn.CrossEntropyLoss()",
optimizer: "torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)",
training_examples: "1000000"
}
};

// Updates page with preset values
function fillPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;

    document.getElementById(module_id).value = preset.module_code.trim();
    document.getElementById(input_shape_id).value = preset.shape_code.trim();
    document.getElementById(loss_fn_id).value = preset.loss_function;
    document.getElementById(optimizer_id).value = preset.optimizer;
    document.getElementById(num_examples_id).value = preset.training_examples;
}