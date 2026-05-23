def main(args: CrossTokenizerDistillArgs):
    logger.info(pformat(args))

    output_dir = Path(args.output)
    # clear previous output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "args.yaml", "w") as f:
        yaml.dump(asdict(args), f)

    # prepare dataset
    dataset = data.get_dataset(**args.data, seed=args.seed)

    teacher_model_kwargs = (
        asdict(args.teacher) if args.teacher is not None else asdict(args.student)
    )
    teacher_tokenizer_name = teacher_model_kwargs.pop("tokenizer_name")
    target_tokenizer_name = args.target_tokenizer_name

    # tokenizer_teacher = load_byteify_tokenizer(teacher_tokenizer_name)
    # target_tokenizer = load_byteify_tokenizer(target_tokenizer_name)

    tokenizer_teacher = AutoTokenizer.from_pretrained(target_tokenizer_name.split(":")[0])
    tokenizer_teacher.pad_token = tokenizer_teacher.eos_token

    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name.split(":")[0])
    target_tokenizer.pad_token = target_tokenizer.eos_token



    new_model = AutoModelForCausalLM.from_pretrained(
        args.student.pretrained_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # max_length=args.max_student_length,
    )

    device = new_model.device

    
    expand_input_ids_dict = None

    collator = TokenizerAlignerCollator(
        tokenizer_teacher,
        target_tokenizer,
        max_teacher_length=args.max_teacher_length,
        max_student_length=args.max_student_length,
        use_chat_template=args.use_chat_template,
        chat_template_mode=args.chat_template_mode,
        expand_input_ids_dict=expand_input_ids_dict,
        loss_mask_mode=args.loss_mask_mode,
        tokenizer_pair_data_path=args.tokenizer_pair_data_path,
        tokenizer_pair_bias_threshold=args.tokenizer_pair_bias_threshold,
        require_bias_matrices=any("unbiased" in x for x in args.losses),
    )

    train_dataloader = DataLoader(
        dataset.get_torch_dataset(),
        batch_size=1,  # batched internally
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=True
    )

    diter = iter(train_dataloader)

    print("train start")

    new_model.train()
    
    optimizer = torch.optim.AdamW(new_model.parameters(), 
                                  lr=args.optimizer['learning_rate'], 
                                #   betas=(args.optimizer['b1'], args.optimizer['b2']), 
                                #   eps=args.optimizer['eps'], 
                                #   weight_decay=args.optimizer['weight_decay']
                                  )
    

    projector_latents = None

    if teacher_model is not None and args.latents_do_project:
        if args.model_type == 'gpt2':
            teacher_hidden_size = teacher_model.config.n_embd
            student_hidden_size = new_model.config.n_embd
        else:
            teacher_hidden_size = teacher_model.config.hidden_size
            student_hidden_size = new_model.config.hidden_size

        projector_latents = torch.nn.Linear(teacher_hidden_size, student_hidden_size)
        projector_latents = projector_latents.to(device)


        optimizer.add_param_group(
            {"params": projector_latents.parameters(), "lr": 2 * args.optimizer['learning_rate']}
        )


    num_training_steps = args.steps
    num_warmup_steps = args.warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for step in tqdm(range(args.steps)):
        try:
            batch = next(diter)
        except StopIteration:
            new_model.save_pretrained(args.output + f"/{step}")
            diter = iter(train_dataloader)
            batch = next(diter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if args.dry_run:
            continue

        optimizer.zero_grad()

        # loss, step_metrics = train_step(batch)
        student_out = new_model(
            input_ids=batch["input_ids_new"],
            attention_mask=batch["attention_mask_new"]
        )
        labels = batch["input_ids_new"].clone().detach()
        labels.masked_fill_(~batch["loss_mask_new"], -100)
        loss = new_model.loss_function(
            student_out.logits,
            labels.view(-1),
            student_out.logits.size(-1)
        )

        loss.backward()
    
        # (Optional) Gradient Clipping
        torch.nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
        
        optimizer.step()
        
        scheduler.step()

        # train_metrics.append(step_metrics)
        if (step + 1) % args.log_interval == 0:
            # print(step_metrics)
            print(loss.item())
            current_lr = scheduler.get_last_lr()[0]
            print("LR:", current_lr)