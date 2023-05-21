import torch
import torch.optim as optim

from DeepRLAgent.SimpleCNNEncoder.Seq2SeqModel import Seq2Seq
from DeepRLAgent.SimpleCNNEncoder.Decoder import Decoder
from DeepRLAgent.SimpleCNNEncoder.Encoder import Encoder
from DeepRLAgent.BaseTrain import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 number_of_index,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 state_mode=1,
                 window_size=1,
                 transaction_cost=0.0,
                 n_classes=64,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is inherited from the BaseTrain class to initialize networks and other stuff that are specific to this
        model. For those parameters in the following explanation that I wrote: "for using in the name of the result file"
        the effect of those parameters has been applied in the Data class and are mentioned here only for begin used as
        part of the experiment's result filename.
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param n_classes: this is the feature vector size of the encoder's output.
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        super(Train, self).__init__(number_of_index, 
                                    data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'SimpleCNNEncoder',
                                    state_mode,
                                    window_size,
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step)
        
        device = 'cpu'
        self.encoder = Encoder(number_of_index, n_classes, data_train.state_size).to(device)
        self.policy_decoder = Decoder(n_classes, 3).to(device)
        self.target_decoder = Decoder(n_classes, 3).to(device)

        self.policy_net = Seq2Seq(self.encoder, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())


        test_encoder = Encoder(number_of_index,n_classes, self.data_train.state_size).to(device)
        test_decoder = Decoder(n_classes, 3).to(device)

        self.test_net = Seq2Seq(test_encoder, test_decoder)
        self.test_net.to(device)
