import torch
import torch.nn as nn
import unittest
from torchviz import make_dot
import os
from torchview import draw_graph

def get_time_embedding(time_steps, t_emb_dim):
    # Calculate the positional frequencies.
    # The division by 2 is to ensure we get pairs for sin and cos functions.
    factor = 10000 ** (torch.arange(start=0, end=t_emb_dim // 2, device=time_steps.device) / (t_emb_dim // 2))
    
    # Calculate the positional encodings.
    # The time_steps[:, None] creates a column vector of time steps, 
    # which is then divided by the factor to adjust the frequencies.
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    
    # Concatenate the sin and cos to get the final time embedding.
    # This ensures that for each frequency, we have a sin and cos component,
    # which helps the model to better distinguish between different positions.
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
    print(t_emb.shape)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample

        # First part of the ResNet block with Group Normalization and SiLU activation
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),  # Group Normalization with 8 groups
            nn.SiLU(),  # SiLU activation function (Swish)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        )

        # Time embedding layers that will be added to the block's output
        self.t_emb_layers = nn.Sequential(
            nn.Linear(t_emb_dim, out_channels),  # Linear layer to match the time embedding dimension to out_channels
            nn.SiLU()  # SiLU activation function
        )

        # Second part of the ResNet block with Group Normalization and SiLU activation
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=out_channels),  # Group Normalization with 8 groups
            nn.SiLU(),  # SiLU activation function
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        )

        # Group Normalization layer before the attention mechanism
        self.attention_norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)

        # Convolutional layer to adjust the number of channels for the residual connection
        self.residul_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Convolutional layer for down-sampling, if enabled; otherwise, an identity mapping
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb):
        # ResNet block

        # Store the original input for the residual connection later
        resnet_input = x

        # Apply the first part of the ResNet block, which includes group normalization, SiLU activation, and a convolutional layer
        out = self.resnet_conv_first(x)  # Corrected from 'out' to 'x'

        # Add the time embedding to the output. The time embedding is expanded to match the spatial dimensions of 'out'
        # The time embedding is added to introduce temporal information into the model
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]

        # Apply the second part of the ResNet block, which further processes the data
        out = self.resnet_conv_second(out)

        # Add the original input (after processing through a convolutional layer if down-sampling is enabled) to the output
        # This forms a residual connection, which helps in training deeper networks by allowing gradients to flow more easily
        out = out + self.residul_input_conv(resnet_input)
        print(out.shape)
        # Attention block

        # Get the shape of the current output to use in reshaping for the attention mechanism
        batch_size, channels, h, w = out.shape

        # Normalize the output before applying attention. This normalization step is crucial for stabilizing the training process
        # Note: The normalization should be applied directly to 'out', not 'in_attn' as initially written
        in_attn = self.attention_norm(out)

        # Reshape the output to a 2D sequence format expected by the multi-head attention mechanism
        # The spatial dimensions (height and width) are flattened into a single dimension
        in_attn = in_attn.reshape(batch_size, channels, h*w)

        # Transpose the dimensions to match the input format expected by nn.MultiheadAttention
        # The sequence length becomes the second dimension, and the channels become the last dimension
        in_attn = in_attn.transpose(1, 2)

        # Apply multi-head self-attention. The same tensor is used as query, key, and value
        # This allows the model to focus on different parts of the input based on the learned attention weights
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)

        # After attention, transpose the dimensions back to their original order and reshape to the original spatial dimensions
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)

        # The output of the attention block is now ready for further processing or to be passed to the next layer
        out_attn = self.down_sample_conv(out_attn)

        return out_attn

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        
        # First set of ResNet blocks with Group Normalization, SiLU activation, and convolution
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        # Time embedding layers to integrate temporal information
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(t_emb_dim, out_channels),
                nn.SiLU()
            )
        ])
        
        # Second set of ResNet blocks
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        # Normalization layer before the attention mechanism
        self.attention_norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        
        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        
        # Convolutional layers for adjusting the number of channels for the residual connection
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])

    def forward(self, x, t_emb):
        out = x
        # Process through the first ResNet block, add time embedding, and apply residual connection
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Normalize and reshape for attention, apply attention, and add the result to the output
        batch_size, channels, h, w = out.shape
        in_attn = self.attention_norm(out)
        in_attn = in_attn.reshape(batch_size, channels, h*w)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # Process through the second ResNet block and apply residual connection
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]  # Note: This should likely be indexed [1] if there's a second t_emb_layer
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=1, out_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=1, out_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention_norm = nn.GroupNorm(num_groups=1, out_channels=out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=4,
                                                 stride=2, padding=1) if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat(tensors=[x, out_down], dim=1)

        # Resnet block
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = self.attention_norm(in_attn)
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = in_attn.tranpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out+out_attn
        
        return out

class TestTimeEmbedding(unittest.TestCase):
    def test_get_time_embedding_shape(self):
        # Set a fixed seed for reproducibility
        torch.manual_seed(42)
        
        # Generate random input arguments
        num_time_steps = torch.randint(low=1, high=100, size=(1,)).item()  # Random number of time steps
        t_emb_dim = torch.randint(low=2, high=512, size=(1,)).item()  # Random embedding dimension (even number)
        if t_emb_dim % 2 != 0:
            t_emb_dim += 1  # Ensure t_emb_dim is even
        
        time_steps = torch.arange(num_time_steps)  # Generate sequential time steps
        
        # Expected output shape
        expected_shape = (num_time_steps, t_emb_dim)
        
        # Call the function
        output = get_time_embedding(time_steps, t_emb_dim)
        
        # Check if the output shape matches the expected shape
        self.assertEqual(output.shape, expected_shape, "Output shape does not match expected shape.")

def viz_downlock():

    model = DownBlock(in_channels=3, out_channels=16, num_heads=4, t_emb_dim=32, down_sample='no')

    # Create dummy tensors
    x = torch.randn(1, 3, 64, 64) # Example input tensor
    t_emb = torch.randn(1, 32) # Example time embedding tensor

    # Generate the visual graph
    model_graph = draw_graph(model, input_data=(x, t_emb))

    # Visualize the graph
    model_graph.visual_graph.render('downblock_architecture', format='png')

def viz_midblock():

    model = MidBlock(in_channels=3, out_channels=16, t_emb_dim=32, num_heads=4)

    # Create dummy input tensors
    x = torch.randn(1, 3, 64, 64)  # Example input tensor
    t_emb = torch.randn(1, 32)  # Example time embedding tensor

    # Generate the visual graph
    model_graph = draw_graph(model, input_data=(x, t_emb))

    # Visualize the graph
    model_graph.visual_graph.render('midblock_architecture', format='png')

# Run the unit tests
if __name__ == '__main__':
    # Call the function to create and save the computational graph
    viz_downlock()
    viz_midblock()
    unittest.main()