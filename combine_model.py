import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, num_classes)

    def forward(self, g, features):
        #64->128->64
        x = self.conv1(g, features)
        h = x
        x = F.relu(x)
        x = self.conv2(g, x)
        return x,h


class CombinedModel(nn.Module):
    def __init__(self, c1, c3, feature_dim, num_classes=2,freeze=True):
        super(CombinedModel, self).__init__()

        # Freeze parameters of the pretrained models
        if freeze==True:
            for param in c1.parameters():
                param.requires_grad = False
            for param in c3.parameters():
                param.requires_grad = False
        else:
            for param in c1.parameters():
                param.requires_grad = True
            for param in c3.parameters():
                param.requires_grad = True

        self.model1 = c1
        self.model2 = c3

        # Logistic regression layer
        self.logistic = nn.Linear(feature_dim, num_classes)

    def forward(self, c1_feat, g_pop):
        # print('length of c1_feat',len(c1_feat))
        #len c1_feat 440

        # Pass the input through the models sequentially
        # Make sure using the right input for each component
        # Process individual embeddings using c1 for each subject
        embeddings = []
        for g_i, feature_i in c1_feat:

            _, embedding = self.model1(g_i, feature_i)
            embedding=torch.max(embedding,dim=0,keepdim=True)[0]

            #embeddings: 379*128 ? should ne 1*128
            embeddings.append(embedding)
            # embeddings(list): 440 {[379*128]}
            # But my c1 embeddings is 440*128, which means this model1 forwarded a wrong size
        # print("c1_feats:", len(c1_feat))
        # print('length of embeddings',len(embeddings))
        # Concatenate individual embeddings along the batch dimension
        population_embedding = torch.cat(embeddings, dim=0)
        # print('length of population embeddings is',population_embedding.size())
        # Pass the aggregated population embedding to c3
        population_output = self.model2(g_pop, population_embedding)
        if isinstance(population_output, tuple):
            population_output = population_output[0]  # Get the first element (tensor)
        # print('population output size is: ',len(population_output))
        # Flatten the output for logistic regression
        x = population_output.view(population_output.size(0), -1)

        # Final prediction through the logistic regression layer
        output = self.logistic(x)
        return output


# Grad-CAM implementation to visualize important regions
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model  # The model we want to interpret
        self.target_layer = target_layer  # The GCN layer we want to target
        self.gradients = None  # To store gradients
        self.activations = None  # To store activations

        # Register hooks to capture activations and gradients during forward/backward passes
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        """Hook to save activations during forward pass."""
        self.activations = output  # Store the activations (node features)

    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        self.gradients = grad_output[0]  # Store the gradients wrt node features

    def __call__(self, c1_feat, g_pop, class_idx=None):
        """
        Forward pass through the model with the graph and features as input.
        - graph: DGL graph object.
        - features: Node feature matrix (tensor).
        - class_idx: Optional index of the class to compute Grad-CAM for.
        """
        # Forward pass through the model
        output = self.model(c1_feat, g_pop)

        print(output.shape)
        # If no class index is provided, use the predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1)
          # Assuming the first element contains the logits

        # Perform backward pass to get gradients
        self.model.zero_grad()
        # output[:, class_idx].backward()
        selected_output = output.gather(1, class_idx.view(-1, 1)).sum()
        selected_output.backward()
        # Pool gradients across the feature dimension (global average pooling)
        pooled_gradients = torch.mean(self.gradients, dim=0)  # [num_features]

        # Weight the activations by the pooled gradients
        weighted_activations = self.activations * pooled_gradients  # Element-wise product

        # Sum across the feature dimension to get node importance scores
        node_importance = weighted_activations.sum(dim=1).detach().cpu().numpy()

        # Normalize node importance scores to [0, 1]
        node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())

        return node_importance  # Return node importance scores


if __name__ == "__main__":
    model1 = torch.load('saved_models_pth/gcn_model_1021_1.pth')
    model2 = torch.load('saved_models_pth/gcn_model_single_graph_site_16_1028_1.pth')

    print(model1)
    print(model2['conv2.bias'].shape)
