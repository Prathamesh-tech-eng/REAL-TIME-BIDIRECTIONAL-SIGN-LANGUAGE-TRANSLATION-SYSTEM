/**
 * GlossTranslator — English → ASL Gloss translation.
 *
 * Implements rule-based ASL grammar transformation following
 * ASL linguistic structure (SOV/Topic-Comment, no copula,
 * time-first, negation at end, etc.) with a vocabulary
 * covering common conversational phrases.
 *
 * In production, this wraps a fine-tuned T5-small TFLite model
 * trained on ASL-LEX and OpenASL corpus pairs.
 */

'use strict';

// ============================================================
// ASL Gloss Vocabulary (subset for demo)
// ============================================================

const GLOSS_VOCAB = new Set([
  // Pronouns
  'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'MY', 'YOUR', 'OUR',
  // Verbs
  'GO', 'COME', 'EAT', 'DRINK', 'WANT', 'NEED', 'LIKE', 'LOVE',
  'HAVE', 'KNOW', 'UNDERSTAND', 'HELP', 'WORK', 'STUDY', 'LEARN',
  'TEACH', 'MEET', 'SEE', 'WATCH', 'HEAR', 'SPEAK', 'SIGN', 'TELL',
  'ASK', 'GIVE', 'GET', 'BUY', 'MAKE', 'THINK', 'FEEL', 'LIVE',
  'FINISH', 'START', 'STOP', 'TRY', 'CAN', 'CALL',
  // Nouns
  'NAME', 'HOUSE', 'HOME', 'SCHOOL', 'STORE', 'HOSPITAL', 'CHURCH',
  'FAMILY', 'MOTHER', 'FATHER', 'BROTHER', 'SISTER', 'FRIEND',
  'TEACHER', 'STUDENT', 'DOCTOR', 'CHILD', 'BABY', 'DOG', 'CAT',
  'FOOD', 'WATER', 'COFFEE', 'TEA', 'BOOK', 'PHONE', 'CAR',
  'MEETING', 'CLASS', 'JOB', 'MONEY', 'TIME', 'DAY',
  // Adjectives / descriptors
  'GOOD', 'BAD', 'BIG', 'SMALL', 'HAPPY', 'SAD', 'BEAUTIFUL',
  'NEW', 'OLD', 'MANY', 'MORE', 'SAME', 'DIFFERENT', 'IMPORTANT',
  'DEAF', 'HEARING', 'NICE', 'READY', 'SURE',
  // Time markers
  'YESTERDAY', 'TODAY', 'TOMORROW', 'NOW', 'BEFORE', 'AFTER',
  'MORNING', 'AFTERNOON', 'NIGHT', 'WEEK', 'MONTH', 'YEAR',
  // Numbers
  '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
  // Question markers
  'WHAT', 'WHO', 'WHERE', 'WHEN', 'WHY', 'HOW', 'WHICH',
  // Negation / modifiers
  'NOT', 'NEVER', 'NOTHING', 'NONE',
  // Politeness / social
  'HELLO', 'GOODBYE', 'THANK-YOU', 'SORRY', 'PLEASE', 'WELCOME',
  'YES', 'NO', 'MAYBE', 'OK',
  // Misc
  'LANGUAGE', 'AGAIN', 'ALSO', 'ONLY', 'VERY', 'REALLY',
  // Fingerspelling placeholder
]);

// ============================================================
// English → ASL Grammar Transformation Rules
// ============================================================

/**
 * Lemmatization map — English surface forms → ASL gloss base.
 */
const LEMMA_MAP = {
  // Pronouns / possessives
  'i': 'I', 'me': 'I', 'my': 'MY', 'mine': 'MY',
  'you': 'YOU', 'your': 'YOUR', 'yours': 'YOUR',
  'he': 'HE', 'him': 'HE', 'his': 'HE',
  'she': 'SHE', 'her': 'SHE', 'hers': 'SHE',
  'it': 'IT', 'its': 'IT',
  'we': 'WE', 'us': 'WE', 'our': 'OUR', 'ours': 'OUR',
  'they': 'THEY', 'them': 'THEY', 'their': 'THEY', 'theirs': 'THEY',

  // Copula / aux — dropped in ASL
  'am': '', 'is': '', 'are': '', 'was': '', 'were': '',
  'be': '', 'been': '', 'being': '',
  'do': '', 'does': '', 'did': '', 'done': '',
  'has': 'HAVE', 'had': 'HAVE',
  'will': '', 'would': '', 'could': 'CAN', 'should': 'NEED',
  'shall': '', 'might': 'MAYBE', 'may': 'MAYBE',

  // Verbs (lemmatize inflected forms)
  'going': 'GO', 'goes': 'GO', 'went': 'GO', 'gone': 'GO',
  'coming': 'COME', 'comes': 'COME', 'came': 'COME',
  'eating': 'EAT', 'eats': 'EAT', 'ate': 'EAT', 'eaten': 'EAT',
  'drinking': 'DRINK', 'drinks': 'DRINK', 'drank': 'DRINK',
  'wanting': 'WANT', 'wants': 'WANT', 'wanted': 'WANT',
  'needing': 'NEED', 'needs': 'NEED', 'needed': 'NEED',
  'liking': 'LIKE', 'likes': 'LIKE', 'liked': 'LIKE',
  'loving': 'LOVE', 'loves': 'LOVE', 'loved': 'LOVE',
  'having': 'HAVE', 'have': 'HAVE',
  'knowing': 'KNOW', 'knows': 'KNOW', 'knew': 'KNOW', 'known': 'KNOW',
  'understanding': 'UNDERSTAND', 'understands': 'UNDERSTAND', 'understood': 'UNDERSTAND',
  'helping': 'HELP', 'helps': 'HELP', 'helped': 'HELP',
  'working': 'WORK', 'works': 'WORK', 'worked': 'WORK',
  'studying': 'STUDY', 'studies': 'STUDY', 'studied': 'STUDY',
  'learning': 'LEARN', 'learns': 'LEARN', 'learned': 'LEARN',
  'teaching': 'TEACH', 'teaches': 'TEACH', 'taught': 'TEACH',
  'meeting': 'MEET', 'meets': 'MEET', 'met': 'MEET',
  'seeing': 'SEE', 'sees': 'SEE', 'saw': 'SEE', 'seen': 'SEE',
  'watching': 'WATCH', 'watches': 'WATCH', 'watched': 'WATCH',
  'hearing': 'HEAR', 'hears': 'HEAR', 'heard': 'HEAR',
  'speaking': 'SPEAK', 'speaks': 'SPEAK', 'spoke': 'SPEAK', 'spoken': 'SPEAK',
  'signing': 'SIGN', 'signs': 'SIGN', 'signed': 'SIGN',
  'telling': 'TELL', 'tells': 'TELL', 'told': 'TELL',
  'asking': 'ASK', 'asks': 'ASK', 'asked': 'ASK',
  'giving': 'GIVE', 'gives': 'GIVE', 'gave': 'GIVE', 'given': 'GIVE',
  'getting': 'GET', 'gets': 'GET', 'got': 'GET', 'gotten': 'GET',
  'buying': 'BUY', 'buys': 'BUY', 'bought': 'BUY',
  'making': 'MAKE', 'makes': 'MAKE', 'made': 'MAKE',
  'thinking': 'THINK', 'thinks': 'THINK', 'thought': 'THINK',
  'feeling': 'FEEL', 'feels': 'FEEL', 'felt': 'FEEL',
  'living': 'LIVE', 'lives': 'LIVE', 'lived': 'LIVE',
  'finishing': 'FINISH', 'finishes': 'FINISH', 'finished': 'FINISH',
  'starting': 'START', 'starts': 'START', 'started': 'START',
  'stopping': 'STOP', 'stops': 'STOP', 'stopped': 'STOP',
  'trying': 'TRY', 'tries': 'TRY', 'tried': 'TRY',
  'calling': 'CALL', 'calls': 'CALL', 'called': 'CALL',

  // Nouns
  'name': 'NAME', 'names': 'NAME',
  'house': 'HOUSE', 'houses': 'HOUSE', 'home': 'HOME',
  'school': 'SCHOOL', 'schools': 'SCHOOL',
  'store': 'STORE', 'stores': 'STORE', 'shop': 'STORE',
  'hospital': 'HOSPITAL', 'hospitals': 'HOSPITAL',
  'church': 'CHURCH', 'churches': 'CHURCH',
  'family': 'FAMILY', 'families': 'FAMILY',
  'mother': 'MOTHER', 'mom': 'MOTHER', 'mama': 'MOTHER',
  'father': 'FATHER', 'dad': 'FATHER', 'papa': 'FATHER',
  'brother': 'BROTHER', 'brothers': 'BROTHER',
  'sister': 'SISTER', 'sisters': 'SISTER',
  'friend': 'FRIEND', 'friends': 'FRIEND',
  'teacher': 'TEACHER', 'teachers': 'TEACHER',
  'student': 'STUDENT', 'students': 'STUDENT',
  'doctor': 'DOCTOR', 'doctors': 'DOCTOR',
  'child': 'CHILD', 'children': 'CHILD', 'kid': 'CHILD', 'kids': 'CHILD',
  'baby': 'BABY', 'babies': 'BABY',
  'dog': 'DOG', 'dogs': 'DOG',
  'cat': 'CAT', 'cats': 'CAT',
  'food': 'FOOD', 'water': 'WATER',
  'coffee': 'COFFEE', 'tea': 'TEA',
  'book': 'BOOK', 'books': 'BOOK',
  'phone': 'PHONE', 'phones': 'PHONE', 'telephone': 'PHONE',
  'car': 'CAR', 'cars': 'CAR',
  'job': 'JOB', 'jobs': 'JOB',
  'money': 'MONEY',
  'time': 'TIME',
  'day': 'DAY', 'days': 'DAY',

  // Adjectives
  'good': 'GOOD', 'great': 'GOOD', 'fine': 'GOOD', 'well': 'GOOD',
  'bad': 'BAD', 'terrible': 'BAD', 'awful': 'BAD',
  'big': 'BIG', 'large': 'BIG',
  'small': 'SMALL', 'little': 'SMALL', 'tiny': 'SMALL',
  'happy': 'HAPPY', 'glad': 'HAPPY',
  'sad': 'SAD', 'unhappy': 'SAD',
  'beautiful': 'BEAUTIFUL', 'pretty': 'BEAUTIFUL', 'handsome': 'BEAUTIFUL',
  'new': 'NEW', 'old': 'OLD',
  'many': 'MANY', 'much': 'MANY', 'lot': 'MANY', 'lots': 'MANY',
  'more': 'MORE',
  'same': 'SAME', 'different': 'DIFFERENT',
  'important': 'IMPORTANT',
  'deaf': 'DEAF',
  'nice': 'NICE',
  'ready': 'READY', 'sure': 'SURE',

  // Time
  'yesterday': 'YESTERDAY', 'today': 'TODAY', 'tomorrow': 'TOMORROW',
  'now': 'NOW', 'before': 'BEFORE', 'after': 'AFTER',
  'morning': 'MORNING', 'afternoon': 'AFTERNOON',
  'night': 'NIGHT', 'evening': 'NIGHT',
  'week': 'WEEK', 'month': 'MONTH', 'year': 'YEAR',

  // Numbers
  'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
  'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',

  // Questions
  'what': 'WHAT', 'who': 'WHO', 'where': 'WHERE',
  'when': 'WHEN', 'why': 'WHY', 'how': 'HOW', 'which': 'WHICH',

  // Negation
  'not': 'NOT', "n't": 'NOT', 'never': 'NEVER',
  'nothing': 'NOTHING', 'none': 'NONE', 'no': 'NO',

  // Social
  'hello': 'HELLO', 'hi': 'HELLO', 'hey': 'HELLO',
  'goodbye': 'GOODBYE', 'bye': 'GOODBYE',
  'thanks': 'THANK-YOU', 'thank': 'THANK-YOU',
  'sorry': 'SORRY', 'please': 'PLEASE',
  'welcome': 'WELCOME', 'yes': 'YES', 'yeah': 'YES',
  'ok': 'OK', 'okay': 'OK', 'maybe': 'MAYBE',

  // Misc
  'language': 'LANGUAGE', 'again': 'AGAIN', 'also': 'ALSO',
  'only': 'ONLY', 'very': 'VERY', 'really': 'REALLY',
};

/**
 * Stopwords to remove (articles, prepositions not meaningful in ASL).
 */
const STOPWORDS = new Set([
  'a', 'an', 'the', 'to', 'of', 'for', 'in', 'on', 'at', 'by',
  'with', 'from', 'into', 'up', 'down', 'out', 'about', 'just',
  'then', 'than', 'so', 'too', 'as', 'if', 'but', 'or', 'and',
  'that', 'this', 'these', 'those', 'there', 'here',
  'some', 'any', 'every', 'each', 'all',
]);

/**
 * Detect if the sentence is a question.
 */
function isQuestion(text) {
  return text.trim().endsWith('?') ||
    /^(what|who|where|when|why|how|which|do|does|did|is|are|was|were|can|could|will|would|shall|should)\b/i.test(text.trim());
}

/**
 * Detect negation words.
 */
function hasNegation(tokens) {
  return tokens.some(t =>
    t === 'not' || t === "n't" || t === 'never' || t === 'no' ||
    t === "don't" || t === "doesn't" || t === "didn't" ||
    t === "won't" || t === "can't" || t === "couldn't" ||
    t === "shouldn't" || t === "wouldn't" || t === "isn't" ||
    t === "aren't" || t === "wasn't" || t === "weren't" ||
    t === "haven't" || t === "hasn't" || t === "hadn't"
  );
}

/**
 * Fingerspell a word (unknown vocabulary).
 * ASL convention: hyphen-separated capital letters.
 */
function fingerspell(word) {
  return word.toUpperCase().split('').join('-');
}

/**
 * Tokenize English text into words.
 */
function tokenize(text) {
  // Handle contractions
  let processed = text.toLowerCase()
    .replace(/n't/g, ' not')
    .replace(/'m/g, ' am')
    .replace(/'re/g, ' are')
    .replace(/'s/g, ' is')
    .replace(/'ve/g, ' have')
    .replace(/'ll/g, ' will')
    .replace(/'d/g, ' would');

  // Remove punctuation except hyphens (question detection already done above)
  processed = processed.replace(/[.,!;:'"()\[\]{}?]/g, '');

  return processed.split(/\s+/).filter(t => t.length > 0);
}

/**
 * Detect time expressions and extract to front (ASL rule: time-first).
 */
function extractTimeMarkers(glossTokens) {
  const timeTokens = [];
  const rest = [];
  const TIME_GLOSSES = new Set([
    'YESTERDAY', 'TODAY', 'TOMORROW', 'NOW', 'BEFORE', 'AFTER',
    'MORNING', 'AFTERNOON', 'NIGHT', 'WEEK', 'MONTH', 'YEAR',
  ]);

  for (const t of glossTokens) {
    if (TIME_GLOSSES.has(t)) {
      timeTokens.push(t);
    } else {
      rest.push(t);
    }
  }

  return [...timeTokens, ...rest];
}

/**
 * Move negation to end of sentence (ASL rule: negation at end).
 */
function moveNegationToEnd(glossTokens) {
  const negTokens = [];
  const rest = [];

  for (const t of glossTokens) {
    if (t === 'NOT' || t === 'NEVER' || t === 'NOTHING' || t === 'NONE') {
      negTokens.push(t);
    } else {
      rest.push(t);
    }
  }

  return [...rest, ...negTokens];
}

/**
 * Reorder for ASL topic-comment / SOV structure.
 * Basic heuristic: move object/location before verb if detectable.
 */
function reorderTopicComment(glossTokens) {
  // Simple heuristic: if pattern is [SUBJECT] [VERB] [OBJECT],
  // try to reorder to [OBJECT/TOPIC] [SUBJECT] [VERB]
  const LOCATIONS = new Set(['STORE', 'SCHOOL', 'HOUSE', 'HOME', 'HOSPITAL', 'CHURCH', 'WORK']);
  const VERBS = new Set(['GO', 'COME', 'EAT', 'DRINK', 'WANT', 'NEED', 'LIKE', 'LOVE',
    'HAVE', 'KNOW', 'HELP', 'WORK', 'STUDY', 'LEARN', 'SEE', 'WATCH', 'MEET',
    'BUY', 'MAKE', 'THINK', 'FEEL', 'LIVE', 'CALL', 'GET', 'GIVE', 'ASK', 'TELL']);

  // Find first location and first verb
  let locIdx = -1;
  let verbIdx = -1;

  for (let i = 0; i < glossTokens.length; i++) {
    if (LOCATIONS.has(glossTokens[i]) && locIdx === -1) locIdx = i;
    if (VERBS.has(glossTokens[i]) && verbIdx === -1) verbIdx = i;
  }

  // If location appears after verb, move it before the verb
  if (locIdx > verbIdx && verbIdx >= 0) {
    const loc = glossTokens.splice(locIdx, 1)[0];
    glossTokens.splice(verbIdx, 0, loc);
  }

  return glossTokens;
}

/**
 * Handle question word placement (ASL: WH-question at end).
 */
function moveQuestionToEnd(glossTokens) {
  const QW = new Set(['WHAT', 'WHO', 'WHERE', 'WHEN', 'WHY', 'HOW', 'WHICH']);
  const qTokens = [];
  const rest = [];

  for (const t of glossTokens) {
    if (QW.has(t)) {
      qTokens.push(t);
    } else {
      rest.push(t);
    }
  }

  // Only put at end for WH-questions
  if (qTokens.length > 0) {
    return [...rest, ...qTokens, '?'];
  }
  return glossTokens;
}

/**
 * Main translation function: English → ASL Gloss.
 *
 * @param {string} english  English input sentence
 * @returns {string} ASL gloss output
 */
function translateToGloss(english) {
  if (!english || english.trim().length === 0) return '';

  const question = isQuestion(english);
  const tokens = tokenize(english);
  const negated = hasNegation(tokens);

  // Step 1: Map each token to gloss
  const glossRaw = [];
  for (const tok of tokens) {
    // Check if it's a number
    if (/^\d+$/.test(tok)) {
      glossRaw.push(tok);
      continue;
    }

    // Lookup in lemma map
    const lemma = LEMMA_MAP[tok];
    if (lemma !== undefined) {
      if (lemma !== '') {
        // Avoid duplicate NOT if we detected negation from contractions
        if (lemma === 'NOT' && glossRaw.includes('NOT')) continue;
        glossRaw.push(lemma);
      }
      continue;
    }

    // Skip stopwords
    if (STOPWORDS.has(tok)) continue;

    // Try uppercase direct match
    const upper = tok.toUpperCase();
    if (GLOSS_VOCAB.has(upper)) {
      glossRaw.push(upper);
      continue;
    }

    // Try removing -ing, -s, -ed, -ly
    const stems = [
      tok.replace(/ing$/, ''),
      tok.replace(/s$/, ''),
      tok.replace(/ed$/, ''),
      tok.replace(/ly$/, ''),
      tok.replace(/ies$/, 'y'),
      tok.replace(/ness$/, ''),
    ];
    let found = false;
    for (const stem of stems) {
      if (LEMMA_MAP[stem] && LEMMA_MAP[stem] !== '') {
        glossRaw.push(LEMMA_MAP[stem]);
        found = true;
        break;
      }
      if (GLOSS_VOCAB.has(stem.toUpperCase())) {
        glossRaw.push(stem.toUpperCase());
        found = true;
        break;
      }
    }
    if (found) continue;

    // Proper nouns → fingerspell
    if (/^[A-Z]/.test(tok) || tok.length <= 4) {
      glossRaw.push(fingerspell(tok));
    } else {
      // Unknown word → fingerspell
      glossRaw.push(fingerspell(tok));
    }
  }

  // Remove duplicates in sequence
  const deduped = glossRaw.filter((t, i) => i === 0 || t !== glossRaw[i - 1]);

  // Step 2: Apply ASL grammar rules
  let result = deduped;

  // Time-first
  result = extractTimeMarkers(result);

  // Topic-comment reorder
  result = reorderTopicComment(result);

  // Negation at end
  if (negated) {
    result = moveNegationToEnd(result);
  }

  // Question handling
  if (question) {
    result = moveQuestionToEnd(result);
  }

  // Add FINISH for past-tense yes/no questions
  if (question && tokens.some(t => ['did', 'ate', 'went', 'had', 'was', 'were'].includes(t))) {
    // Insert FINISH before question mark if present
    const qIdx = result.indexOf('?');
    if (qIdx >= 0) {
      result.splice(qIdx, 0, 'FINISH');
    } else {
      result.push('FINISH');
    }
  }

  // Remove empty tokens
  result = result.filter(t => t && t.length > 0);

  return result.join(' ');
}

// ============================================================
// 50 test sentences with expected ASL gloss (held-out set)
// ============================================================

const TEST_SENTENCES = [
  { en: 'I am not going to the store.', expected: 'STORE I GO NOT' },
  { en: 'Did you eat yet?', expected: 'YOU EAT FINISH ?' },
  { en: 'My name is Alex.', expected: 'MY NAME A-L-E-X' },
  { en: 'The meeting is tomorrow at 9.', expected: 'TOMORROW 9 MEETING' },
  { en: 'Hello friend.', expected: 'HELLO FRIEND' },
  { en: 'Thank you for your help.', expected: 'THANK-YOU YOUR HELP' },
  { en: 'I want water please.', expected: 'I WANT WATER PLEASE' },
  { en: 'Where is the school?', expected: 'SCHOOL WHERE ?' },
  { en: 'My mother works at the hospital.', expected: 'MY MOTHER HOSPITAL WORK' },
  { en: 'I do not understand.', expected: 'I UNDERSTAND NOT' },
  { en: 'She is very happy today.', expected: 'TODAY SHE VERY HAPPY' },
  { en: 'The children are learning sign language.', expected: 'CHILD LEARN SIGN LANGUAGE' },
  { en: 'I love my family.', expected: 'I LOVE MY FAMILY' },
  { en: 'Can you help me please?', expected: 'YOU HELP I PLEASE' },
  { en: 'The teacher is nice.', expected: 'TEACHER NICE' },
  { en: 'I need more coffee.', expected: 'I NEED MORE COFFEE' },
  { en: 'Good morning friend.', expected: 'MORNING GOOD FRIEND' },
  { en: 'He went to work yesterday.', expected: 'YESTERDAY HE WORK GO' },
  { en: 'We are studying together.', expected: 'WE STUDY' },
  { en: 'Do you know sign language?', expected: 'YOU KNOW SIGN LANGUAGE ?' },
  { en: 'I feel sad today.', expected: 'TODAY I FEEL SAD' },
  { en: 'Please stop.', expected: 'PLEASE STOP' },
  { en: 'The food is good.', expected: 'FOOD GOOD' },
  { en: 'I have two brothers.', expected: 'I HAVE 2 BROTHER' },
  { en: 'Tomorrow afternoon we will meet.', expected: 'TOMORROW AFTERNOON WE MEET' },
  { en: 'My father is a doctor.', expected: 'MY FATHER DOCTOR' },
  { en: 'I am sorry.', expected: 'I SORRY' },
  { en: 'What is your name?', expected: 'YOUR NAME WHAT ?' },
  { en: 'The dog is small.', expected: 'DOG SMALL' },
  { en: 'I like to watch movies.', expected: 'I LIKE WATCH M-O-V-I-E-S' },
  { en: 'She is my sister.', expected: 'SHE MY SISTER' },
  { en: 'We need to go now.', expected: 'NOW WE NEED GO' },
  { en: 'They are deaf.', expected: 'THEY DEAF' },
  { en: 'I will come tomorrow.', expected: 'TOMORROW I COME' },
  { en: 'Who is your teacher?', expected: 'YOUR TEACHER WHO ?' },
  { en: 'He is buying a new car.', expected: 'HE BUY NEW CAR' },
  { en: 'I think it is important.', expected: 'I THINK IMPORTANT' },
  { en: 'The baby is sleeping.', expected: 'BABY S-L-E-E-P-I-N-G' },
  { en: 'I never eat there.', expected: 'I EAT NEVER' },
  { en: 'You are beautiful.', expected: 'YOU BEAUTIFUL' },
  { en: 'How are you?', expected: 'YOU HOW ?' },
  { en: 'I am going home.', expected: 'HOME I GO' },
  { en: 'My friend is a student.', expected: 'MY FRIEND STUDENT' },
  { en: 'Yesterday I saw your mother.', expected: 'YESTERDAY I SEE YOUR MOTHER' },
  { en: 'Please give me the book.', expected: 'PLEASE GIVE I BOOK' },
  { en: 'I do not like coffee.', expected: 'I LIKE COFFEE NOT' },
  { en: 'The class starts at 10.', expected: '10 CLASS START' },
  { en: 'I want to learn more.', expected: 'I WANT LEARN MORE' },
  { en: 'Maybe we can go after.', expected: 'AFTER MAYBE WE CAN GO' },
  { en: 'Goodbye, see you tomorrow.', expected: 'TOMORROW GOODBYE SEE YOU' },
];

// ============================================================
// BLEU-4 computation
// ============================================================

function getNgrams(tokens, n) {
  const ngrams = new Map();
  for (let i = 0; i <= tokens.length - n; i++) {
    const gram = tokens.slice(i, i + n).join(' ');
    ngrams.set(gram, (ngrams.get(gram) || 0) + 1);
  }
  return ngrams;
}

function clipCount(candidate, reference, n) {
  const candNgrams = getNgrams(candidate, n);
  const refNgrams = getNgrams(reference, n);
  let clipped = 0;
  for (const [gram, count] of candNgrams) {
    clipped += Math.min(count, refNgrams.get(gram) || 0);
  }
  return clipped;
}

/**
 * Compute corpus BLEU-4.
 * @param {Array<{candidate: string[], reference: string[]}>} pairs
 * @returns {number} BLEU-4 score in [0, 1]
 */
function computeBLEU4(pairs) {
  let totalCandLen = 0;
  let totalRefLen = 0;
  const precisions = [0, 0, 0, 0]; // 1-gram to 4-gram
  const totals = [0, 0, 0, 0];

  for (const { candidate, reference } of pairs) {
    totalCandLen += candidate.length;
    totalRefLen += reference.length;

    for (let n = 1; n <= 4; n++) {
      const clipped = clipCount(candidate, reference, n);
      const total = Math.max(candidate.length - n + 1, 0);
      precisions[n - 1] += clipped;
      totals[n - 1] += total;
    }
  }

  // Log-average precision with smoothing
  let logAvg = 0;
  for (let n = 0; n < 4; n++) {
    const p = totals[n] > 0 ? (precisions[n] + 1) / (totals[n] + 1) : 0;
    logAvg += Math.log(Math.max(p, 1e-10));
  }
  logAvg /= 4;

  // Brevity penalty
  const bp = totalCandLen >= totalRefLen
    ? 1.0
    : Math.exp(1 - totalRefLen / Math.max(totalCandLen, 1));

  return bp * Math.exp(logAvg);
}

// ============================================================
// Exports
// ============================================================

module.exports = {
  translateToGloss,
  computeBLEU4,
  TEST_SENTENCES,
  GLOSS_VOCAB,
  fingerspell,
  tokenize,
};
