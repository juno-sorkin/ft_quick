outline

objective: finetune unsloth/gpt-oss-20b for text style mimicry

will need to pull model arch as its moe to use correct module names
project specs:
    ci/cd flow:
        docker
        python
        micromamba in image running base
        runner: github actions, on push 
        workflows
            runs on push/merge via actions
            btr.yml = build test release, 
                first pulls image cache from ecr (will need to auth) to prevent rebuilding image every ci
                builds docker image, 
                runs local test suite via pytest and tests dir
                    cpu smoke test to make sure model runs
                conditional on success: release to aws ECR 
            deploy.yml
                on manual activation via cli
                starts ec2 instance via asg ( will have to assume role and auth)
                waits, then SSM's in,
                    runs quick smoke test for nvidia smi and cuda presence
                runs --gpus all with env vars injected (wandb)
                    once running, container should run entrypoint (this is where we choose whether we are running via passed mode on github actions trigger)
                    which will pull training + val, or test (if test run) from s3 bucket
                    runs main training script (or test)
                    saves data (logs, adapters, checkpoints etc) to s3: bucket once done (this should all be done via saving to local paths in the container then syncing to s3 bucket, IE copying into or out of bucket)
                shutsdown ec2 instance via asg (decreasign desire capcity to 0 )

more specs:
    peft lora
    hugging face ecosystem like  datasets, traienr etc
    unsloth
    must use perplexity for loss curve (obviously)
    pytest
    pytorch
    wandb
    docker
    aws - asg, ec2, s3, auth, ssm, roles to assume
    secrets store in env vars + github repo secrets to be injected when needed 
    anything else: ASK for clarification, don't assume.
    


                
                

                
