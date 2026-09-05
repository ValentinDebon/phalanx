/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <err.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <llama.h>

enum phalanx_state {
	PHALANX_STATE_INACTIVE,
	PHALANX_STATE_INPUT_PROMPT,
	PHALANX_STATE_TOKENS_DECODE,
	PHALANX_STATE_OUTPUT_ANALYZE,
};

enum phalanx_dialogue {
	PHALANX_DIALOGUE_TEXT_RESPONSE,
	PHALANX_DIALOGUE_TOOL_CALLS,
	PHALANX_DIALOGUE_ARGS,
};

struct phalanx {
	enum phalanx_state state;
	enum phalanx_dialogue dialogue;

	const struct llama_vocab *vocab;
	struct llama_context *context;
	struct llama_sampler *sampler;

	llama_token *tokens;
	size_t tokens_length;
	size_t tokens_capacity;

	FILE *function_name;
	char *function_name_buffer;
	size_t function_name_size;

	FILE *function_results;
	char *function_results_buffer;
	size_t function_results_size;

	int function_output_rd;
	int function_input_rd;
	FILE *function_input;
	pid_t function_pid;

	char *config_path;
	char *data_path;
};

/***********
 * Private *
 ***********/

#define PHALANX_TOKEN_BOS 1
#define PHALANX_TOKEN_EOG 2
#define PHALANX_TOKEN_INST_BEGIN 3
#define PHALANX_TOKEN_INST_END 4
#define PHALANX_TOKEN_AVAILABLE_TOOLS_BEGIN 5
#define PHALANX_TOKEN_AVAILABLE_TOOLS_END 6
#define PHALANX_TOKEN_TOOL_RESULTS_BEGIN 7
#define PHALANX_TOKEN_TOOL_RESULTS_END 8
#define PHALANX_TOKEN_TOOL_CALLS 9
#define PHALANX_TOKEN_SYSTEM_PROMPT_BEGIN 17
#define PHALANX_TOKEN_SYSTEM_PROMPT_END 18
#define PHALANX_TOKEN_ARGS 32

static void
vocab_token_print(const struct llama_vocab *vocab, llama_token token, FILE *output) {
	int32_t n = -128;

	do {
		char buf[-n];
		n = llama_token_to_piece(vocab,
			token, buf, sizeof (buf), 0, true);
		if (n >= 0) {
			fwrite(buf, sizeof (*buf), n, output);
		}
	} while (n < 0);
}

static void
phalanx_tokens_headroom(struct phalanx *phalanx,
	llama_token **tokensp, size_t *lengthp) {
	*tokensp = phalanx->tokens + phalanx->tokens_length;
	*lengthp = phalanx->tokens_capacity - phalanx->tokens_length;
}

static void
phalanx_tokens_reserve(struct phalanx *phalanx, size_t capacity) {
	phalanx->tokens = realloc(phalanx->tokens,
		capacity * sizeof (*phalanx->tokens));
	phalanx->tokens_capacity = capacity;
}

static void
phalanx_tokens_append(struct phalanx *phalanx, llama_token token) {
	if (phalanx->tokens_length == phalanx->tokens_capacity) {
		phalanx_tokens_reserve(phalanx, phalanx->tokens_capacity + 1);
	}
	phalanx->tokens[phalanx->tokens_length++] = token;
}

static void
phalanx_tokens_feedback(struct phalanx *phalanx, llama_token token) {
	assert(phalanx->tokens_capacity >= 1);
	phalanx->tokens[0] = token;
	phalanx->tokens_length = 1;
}

static void
phalanx_tokenize(struct phalanx *phalanx,
	const char *prompt, size_t prompt_length) {
	llama_token *tokens;
	size_t tokens_length;
	int32_t count;

	while (phalanx_tokens_headroom(phalanx, &tokens, &tokens_length),
		(count = llama_tokenize(phalanx->vocab,
			prompt, prompt_length, tokens, tokens_length, false, true)) < 0) {
		phalanx_tokens_reserve(phalanx, phalanx->tokens_length + -count);
	}

	phalanx->tokens_length += count;
}

static void
phalanx_activate_system_prompt(struct phalanx *phalanx) {
	static const char phalanx_system_prompt[] =
		"You are Phalanx. Follow the user's instructions exactly."
		"Output only the requested result. Do not explain unless explicitly asked.";

	phalanx_tokens_append(phalanx, PHALANX_TOKEN_BOS);
	phalanx_tokens_append(phalanx, PHALANX_TOKEN_SYSTEM_PROMPT_BEGIN);
	phalanx_tokenize(phalanx, phalanx_system_prompt, sizeof (phalanx_system_prompt) - 1);
	phalanx_tokens_append(phalanx, PHALANX_TOKEN_SYSTEM_PROMPT_END);
}

static void
phalanx_activate_available_tools(struct phalanx *phalanx) {
	char *path;

	asprintf(&path, "%s/available_tools.json", phalanx->config_path);

	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		goto open_failure;
	}

	struct stat st;
	if (fstat(fd, &st) < 0 || st.st_size == 0) {
		goto stat_failure;
	}

	void * const address = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (address == MAP_FAILED) {
		goto mmap_failure;
	}

	phalanx_tokens_append(phalanx, PHALANX_TOKEN_AVAILABLE_TOOLS_BEGIN);
	phalanx_tokenize(phalanx, address, st.st_size);
	phalanx_tokens_append(phalanx, PHALANX_TOKEN_AVAILABLE_TOOLS_END);

	munmap(address, st.st_size);
mmap_failure:
stat_failure:
	close(fd);
open_failure:
	free(path);
}

static void
phalanx_function_spawn(struct phalanx *phalanx) {

	fflush(phalanx->function_name);

	int input[2];
	if (pipe2(input, O_CLOEXEC) < 0) {
		err(EXIT_FAILURE, "pipe2 input");
	}

	int output[2];
	if (pipe2(output, O_CLOEXEC) < 0) {
		err(EXIT_FAILURE, "pipe2 output");
	}

	const pid_t pid = fork();
	if (pid < 0) {
		err(EXIT_FAILURE, "fork");
	}

	if (pid == 0) {
		char * const name = phalanx->function_name_buffer;

		if (*name == '\0' || *name == '.'
			|| strchr(name, '/') != NULL) {
			warnx("Invalid tool name '%s'", name);
			_Exit(-1);
		}

		if (dup3(input[0], STDIN_FILENO, 0) < 0) {
			warn("dup3 STDIN_FILENO");
			_Exit(-1);
		}

		if (dup3(output[1], STDOUT_FILENO, 0) < 0) {
			warn("dup3 STDOUT_FILENO");
			_Exit(-1);
		}

		char *path;
		if (asprintf(&path, "%s/%s", phalanx->data_path, name) < 0) {
			warn("asprintf");
			_Exit(-1);
		}

		char * const arguments[] = { name, NULL, };
		execv(path, arguments);

		warn("execv %s (%s)", path, name);
		_Exit(-1);
	}

	close(output[1]);

	phalanx->function_output_rd = output[0];
	phalanx->function_input_rd = input[0];
	phalanx->function_input = fdopen(input[1], "w");
	phalanx->function_pid = pid;
}

static void
phalanx_function_results_wait(struct phalanx *phalanx) {
	char buf[512];
	ssize_t copied;

	fclose(phalanx->function_input);
	close(phalanx->function_input_rd);

	rewind(phalanx->function_results);

	while ((copied = read(phalanx->function_output_rd, buf, sizeof (buf))) > 0) {
		fwrite(buf, sizeof (*buf), copied, phalanx->function_results);
	}

	if (copied < 0) {
		err(EXIT_FAILURE, "read");
	}

	close(phalanx->function_output_rd);

	int wstatus;
	if (waitpid(phalanx->function_pid, &wstatus, 0) < 0) {
		err(EXIT_FAILURE, "waitpid");
	}

	fflush(phalanx->function_results);

	phalanx_tokens_append(phalanx, PHALANX_TOKEN_TOOL_RESULTS_BEGIN);
	phalanx_tokenize(phalanx, phalanx->function_results_buffer, phalanx->function_results_size);
	phalanx_tokens_append(phalanx, PHALANX_TOKEN_TOOL_RESULTS_END);
}

static void
phalanx_dialogue_text_response(struct phalanx *phalanx, llama_token token) {
	if (token != PHALANX_TOKEN_EOG) {
		if (token != PHALANX_TOKEN_TOOL_CALLS) {
			vocab_token_print(phalanx->vocab, token, stdout);
		} else {
			phalanx->dialogue = PHALANX_DIALOGUE_TOOL_CALLS;
			rewind(phalanx->function_name);
		}
		phalanx_tokens_feedback(phalanx, token);
		phalanx->state = PHALANX_STATE_TOKENS_DECODE;
	} else {
		fputc('\n', stdout);
		phalanx->state = PHALANX_STATE_INPUT_PROMPT;
		phalanx->dialogue = PHALANX_DIALOGUE_TEXT_RESPONSE;
	}
}

static void
phalanx_dialogue_tool_calls(struct phalanx *phalanx, llama_token token) {

	switch (token) {
	case PHALANX_TOKEN_EOG:
		phalanx_function_spawn(phalanx);
		phalanx_function_results_wait(phalanx);
		phalanx->dialogue = PHALANX_DIALOGUE_TEXT_RESPONSE;
		break;
	case PHALANX_TOKEN_ARGS:
		phalanx_function_spawn(phalanx);
		phalanx_tokens_feedback(phalanx, token);
		phalanx->dialogue = PHALANX_DIALOGUE_ARGS;
		break;
	default:
		vocab_token_print(phalanx->vocab, token, phalanx->function_name);
		phalanx_tokens_feedback(phalanx, token);
		break;
	}

	phalanx->state = PHALANX_STATE_TOKENS_DECODE;
}

static void
phalanx_dialogue_args(struct phalanx *phalanx, llama_token token) {

	switch (token) {
	case PHALANX_TOKEN_EOG:
		phalanx_function_results_wait(phalanx);
		phalanx->dialogue = PHALANX_DIALOGUE_TEXT_RESPONSE;
		break;
	case PHALANX_TOKEN_TOOL_CALLS:
		phalanx_function_results_wait(phalanx);
		phalanx_tokens_feedback(phalanx, token);
		phalanx->dialogue = PHALANX_DIALOGUE_TOOL_CALLS;
		rewind(phalanx->function_name);
		break;
	default:
		vocab_token_print(phalanx->vocab, token, phalanx->function_input);
		phalanx_tokens_feedback(phalanx, token);
		break;
	}

	phalanx->state = PHALANX_STATE_TOKENS_DECODE;
}

/**********
 * Public *
 **********/

static void
phalanx_create(struct phalanx **phalanxp) {
	struct phalanx * const phalanx = malloc(sizeof (*phalanx));

	phalanx->state = PHALANX_STATE_INACTIVE;
	phalanx->dialogue = PHALANX_DIALOGUE_TEXT_RESPONSE;

	phalanx->tokens = NULL;
	phalanx->tokens_length = 0;
	phalanx->tokens_capacity = 0;

	phalanx->function_name = open_memstream(&phalanx->function_name_buffer, &phalanx->function_name_size);
	phalanx->function_results = open_memstream(&phalanx->function_results_buffer, &phalanx->function_results_size);

	const char * const home = getenv("HOME");
	if (home == NULL || *home != '/') {
		abort();
	}

	const char * const xdg_config_home = getenv("XDG_CONFIG_HOME");
	if (xdg_config_home != NULL && *xdg_config_home == '/') {
		asprintf(&phalanx->config_path, "%s/phalanx", xdg_config_home);
	} else {
		asprintf(&phalanx->config_path, "%s/.config/phalanx", home);
	}

	const char * const xdg_data_home = getenv("XDG_DATA_HOME");
	if (xdg_data_home != NULL && *xdg_data_home == '/') {
		asprintf(&phalanx->data_path, "%s/phalanx", xdg_data_home);
	} else {
		asprintf(&phalanx->data_path, "%s/.local/share/phalanx", home);
	}

	*phalanxp = phalanx;
}

static void
phalanx_destroy(struct phalanx *phalanx) {
	free(phalanx->tokens);

	fclose(phalanx->function_name);
	free(phalanx->function_name_buffer);

	fclose(phalanx->function_results);
	free(phalanx->function_results_buffer);

	if (phalanx->dialogue == PHALANX_DIALOGUE_ARGS) {
		if (kill(phalanx->function_pid, SIGTERM) < 0) {
			err(EXIT_FAILURE, "kill");
		}

		fclose(phalanx->function_input);
		close(phalanx->function_input_rd);
		close(phalanx->function_output_rd);

		if (waitpid(phalanx->function_pid, NULL, 0) < 0) {
			err(EXIT_FAILURE, "waitpid");
		}
	}

	free(phalanx->config_path);
	free(phalanx->data_path);

	free(phalanx);
}

static void
phalanx_activate(struct phalanx *phalanx,
	struct llama_context *context, struct llama_sampler *sampler) {

	if (phalanx->state != PHALANX_STATE_INACTIVE) {
		return;
	}

	const struct llama_model * const model = llama_get_model(context);
	phalanx->vocab = llama_model_get_vocab(model);
	phalanx->context = context;
	phalanx->sampler = sampler;

	phalanx_activate_system_prompt(phalanx);
	phalanx_activate_available_tools(phalanx);

	phalanx->state = PHALANX_STATE_INPUT_PROMPT;
}

static void
phalanx_input_prompt(struct phalanx *phalanx, const char *prompt, size_t length) {

	if (phalanx->state != PHALANX_STATE_INPUT_PROMPT) {
		return;
	}

	phalanx_tokens_append(phalanx, PHALANX_TOKEN_INST_BEGIN);
	phalanx_tokenize(phalanx, prompt, length);
	phalanx_tokens_append(phalanx, PHALANX_TOKEN_INST_END);

	phalanx->state = PHALANX_STATE_TOKENS_DECODE;
}

static void
phalanx_tokens_decode(struct phalanx *phalanx) {

	if (phalanx->state != PHALANX_STATE_TOKENS_DECODE) {
		return;
	}

	const struct llama_batch batch = llama_batch_get_one(phalanx->tokens, phalanx->tokens_length);
	const int32_t n = llama_decode(phalanx->context, batch);
	if (n != 0) {
		abort();
	}

	phalanx->state = PHALANX_STATE_OUTPUT_ANALYZE;
}

static bool
phalanx_output_analyze(struct phalanx *phalanx) {

	if (phalanx->state == PHALANX_STATE_OUTPUT_ANALYZE) {
		const llama_token token = llama_sampler_sample(phalanx->sampler, phalanx->context, -1);

		switch (phalanx->dialogue) {
		case PHALANX_DIALOGUE_TEXT_RESPONSE:
			phalanx_dialogue_text_response(phalanx, token);
			break;
		case PHALANX_DIALOGUE_TOOL_CALLS:
			phalanx_dialogue_tool_calls(phalanx, token);
			break;
		case PHALANX_DIALOGUE_ARGS:
			phalanx_dialogue_args(phalanx, token);
			break;
		}
	}

	return phalanx->state == PHALANX_STATE_TOKENS_DECODE;
}

static void
print_error_log(enum ggml_log_level level, const char *text, void *user_data) {
	FILE * const output = user_data;

	if (level >= GGML_LOG_LEVEL_ERROR) {
		fputs(text, output);
	}
}

int
main(int argc, char *argv[]) {
	struct phalanx *phalanx;

	phalanx_create(&phalanx);

	llama_log_set(print_error_log, stderr);
	ggml_backend_load_all();

	const char * const model_path = CONFIG_LARGE_LANGUAGE_MODEL_PATH;
	struct llama_model_params model_params = llama_model_default_params();
	struct llama_model * const model = llama_model_load_from_file(model_path, model_params);
	if (model == NULL) {
		errx(EXIT_FAILURE, "Unable to load model '%s'", model_path);
	}

	struct llama_context_params context_params = llama_context_default_params();
	context_params.n_ctx = 4096;
	context_params.n_batch = 4096;
	struct llama_context * const context = llama_init_from_model(model, context_params);
	if (context == NULL) {
		errx(EXIT_FAILURE, "Unable to create context from model");
	}

	struct llama_sampler_chain_params sampler_chain_params = llama_sampler_chain_default_params();
	struct llama_sampler * const sampler = llama_sampler_chain_init(sampler_chain_params);
	llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.01f));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	phalanx_activate(phalanx, context, sampler);

	size_t n;
	ssize_t length;
	char *line = NULL;
	while (errno = 0, length = getline(&line, &n, stdin), length > 0) {
		phalanx_input_prompt(phalanx, line, length);
		do {
			phalanx_tokens_decode(phalanx);
		} while (phalanx_output_analyze(phalanx));
	}

	if (length < 0 && errno != 0) {
		err(EXIT_FAILURE, "getline");
	}

	free(line);

	llama_sampler_free(sampler);
	llama_free(context);

	llama_model_free(model);

	phalanx_destroy(phalanx);

	return EXIT_SUCCESS;
}
