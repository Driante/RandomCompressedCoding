using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
include(srcdir("network.jl"))
include(srcdir("decoder.jl"))

function compare_decoders(n)
    n = Network(N,L,σi,rxrnorm=0);
    W = n.W
    for σi = σVec
        n2 = Network(N,L,σi,f1=n.f1,rxrnorm=0);
        n2.W = W
        data_trn,data_tst = iidgaussian_dataset(n2,η,onehot=true);
        λ_id,b_id = ideal_decoder_iidgaussian(n2,η,x_test)
        ce_id,mse_id = ideal_loss(n2,η,x_test,data_tst);

        prob_decoder,history_prob = train_prob_decoder((data_trn,data_tst),epochs=50);

        mlp_decoder,history_mlp = train_mlp_decoder((data_trn,data_tst),epochs=50,M=M);
        push!(mseVec,mse_id)
        push!(probhVec,history_prob)
        push!(mlphVec,history_mlp)
        @info "σ = $σi"
    end
