import React, { useState } from 'react'
import { Grid, Form, TextArea, Dropdown, Button, Container, Message, Label, Input, Icon, Radio, Tab, Table, Accordion, Popup, Segment, Image } from 'semantic-ui-react'
import api from './api'
import RELATIONS from './relations'
import _ from 'lodash'
import {CopyToClipboard} from 'react-copy-to-clipboard'
import {saveAs} from 'file-saver'
import NumberInput from 'semantic-ui-react-numberinput';

import './App.css'

function App() {
  const modelOptions = [
    // {
    //   key: 'comet-gpt2',
    //   text: 'COMET-GPT2',
    //   value: 'comet-gpt2'
    // },
    {
      key: 'comet-bart',
      text: 'COMET-BART',
      value: 'comet-bart'
    },
    // {
    //   key: 'gpt2',
    //   text: 'GPT-2',
    //   value: 'gpt2'
    // }
  ]

  const relationOptions = _.map(RELATIONS, rel => {
    return {key: rel, text: rel, value: rel}
  })

  const headProcOptions = [
    {
      key: 'sentence_extractor',
      text: 'Sentence Extractor',
      value: 'sentence_extractor'
    },
    {
      key: 'noun_phrase_extractor',
      text: 'Noun Phrase Extractor',
      value: 'noun_phrase_extractor'
    },
    {
      key: 'verb_phrase_extractor',
      text: 'Verb Phrase Extractor',
      value: 'verb_phrase_extractor'
    }
  ]

  const relProcOptions = [
    {
      key: 'simple_relation_matcher',
      text: 'Heuristic Matcher',
      value: 'simple_relation_matcher'
    },
    // {
    //   key: 'swem_relation_matcher',
    //   text: 'GloVe-based Matcher',
    //   value: 'swem_relation_matcher'
    // },
    {
      key: 'distilbert_relation_matcher',
      text: 'DistilBERT-based Matcher',
      value: 'distilbert_relation_matcher'
    },
    // {
    //   key: 'bert_relation_matcher',
    //   text: 'BERT-based Matcher',
    //   value: 'bert_relation_matcher'
    // },
  ]

  const [text, setText] = useState('')
  const [model, setModel] = useState('comet-bart')
  const [results, setResults] = useState('')
  const [heads, setHeads] = useState([])
  const [relations, setRelations] = useState([])
  const [extractHeads, setExtractHeads] = useState(true)
  const [matchRelations, setMatchRelations] = useState(true)
  const [dryRun, setDryRun] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [showError, setShowError] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [headProcs, setHeadProcs] = useState(['sentence_extractor', 'noun_phrase_extractor', 'verb_phrase_extractor'])
  const [relProcs, setRelProcs] = useState(['simple_relation_matcher'])
  const [activeHead, setActiveHead] = useState(null)
  const [copiedResults, setCopiedResults] = useState(false)
  const [showNote, setShowNote] = useState(false)
  const [context, setContext] = useState('')
  const [threshold, setThreshold] = useState(0.5)
  const [textWords, setTextWords] = useState([])
  const [inferenceFiltering, setInferenceFiltering] = useState(false)
  const [resultWarningVisible, setResultWarningVisible] = useState(true)

  let resultMap = {}

  for (let res of results) {
    if (!_.has(resultMap, res['head'])) {
      resultMap[res['head']] = []
    }
    resultMap[res['head']].push({relation: res['relation'], tails: res['tails']})
  }

  const orderByFirstWord = o => {
    let words = _.split(o[0], " ")

    if (!_.isEmpty(textWords)) {
      let indices = []

      for (let word of words) {
        let index = _.indexOf(textWords, _.toLower(word))
        if (index !== -1) indices.push(index)
      }

      if (!_.isEmpty(indices)) return _.min(indices)
    }

    return -1
  }

  resultMap = _.sortBy(_.toPairs(resultMap), [orderByFirstWord, o => -_.size(_.split(o[0], " ")), o => _.toLower(o[0])])

  const generate = () => {
    setGenerating(true)
    setShowError(false)
    setShowNote(false)

    if (_.includes(model, "gpt2")) {
      setTimeout(() => {
        setShowNote(true)
      }, 1000)
    }

    let data = {
      text: text,
      model: model,
      heads: _.filter(heads, h => !_.isEmpty(_.trim(h))),
      relations: relations,
      extractHeads: extractHeads,
      matchRelations: matchRelations,
      dryRun: dryRun,
      headProcs: headProcs,
      relProcs: relProcs,
      threshold: threshold
    }

    if (inferenceFiltering) {
      data[context] = context || text
    }

    api.inference
    .generate(data)
    .then(response => {
      setResults(response.data.graph)
      setTextWords(response.data.text)
      setGenerating(false)
      setShowNote(false)
    })
    .catch(error => {
      setGenerating(false)
      setErrorMsg(error.response.data)
      setShowError(true)
      setShowNote(false)
    })
  }

  const handleHeadChange = (index, event) => {
    let newHeads = [...heads]
    newHeads[index] = event.target.value
    setHeads(newHeads)
  }

  const removeHead = (index) => {
    let newHeads = [...heads]
    _.pullAt(newHeads, index)
    setHeads(newHeads)
  }

  const addHead = () => {
    setHeads([...heads, ''])
  }

  const clearResults = () => {
    setResults('')
  }

  const saveResults = () => {
    if (!_.isEmpty(results)) {
      const resultsFile = new Blob([JSON.stringify(results, null, 4)], {type: "text/json;charset=utf-8"})
      saveAs(
        resultsFile,
        "kogito-results.json"
      )
    }
  }

  const copyResults = () => {
    if (!_.isEmpty(results)) {
      setCopiedResults(true)
      setTimeout(() => {
        setCopiedResults(false)
      }, 3000)
    }
  }

  const getHeadsJSX = () => {
    return _.isEmpty(heads) ? null :
      <Container>
        {_.map(heads, (head, index) => {
          return (
            <Container key={index} className='cntr-head'>
              <Input
                fluid
                placeholder='Head text'
                onChange={e => handleHeadChange(index, e)}
                value={head}
                label={<Button icon onClick={e => removeHead(index)}><Icon name='minus'></Icon></Button>}
                labelPosition='right'
              />
            </Container>
          )
        })}
      </Container>
  }

  const resultJSONPane = (
    <div>
      <Segment attached basic className='home-results-json-segment'>
        <Form>
          <TextArea
            placeholder='Results'
            value={_.isEmpty(results) ? '' : JSON.stringify(results, null, 4)}
            rows={30}
            disabled/>
        </Form>
      </Segment>
      <Button.Group basic size='small' attached="bottom">
        <Button icon='trash' onClick={clearResults}/>
        <CopyToClipboard text={_.isEmpty(results) ? '' : JSON.stringify(results, null, 4)}
          onCopy={copyResults}>
          <Button icon='copy'/>
        </CopyToClipboard>
        <Button icon='download' onClick={saveResults}/>
      </Button.Group>
      <Message attached='bottom' success hidden={!copiedResults}>Copied!</Message>
    </div>
  )

  const handleActiveHeadChange = (e, data) => {
    if (activeHead === data.index) {
      return setActiveHead(null)
    }
    return setActiveHead(data.index)
  }

  const resultTablePane = () => {
    return _.map(resultMap, result => {
      let head = result[0]
      let headResults = result[1]
      return (
        <Grid key={head}>
          <Grid.Row>
            <Grid.Column>
              <Accordion styled fluid>
                <Accordion.Title active={activeHead === head} index={head} onClick={(e, data) => handleActiveHeadChange(e, data)}>
                  <Icon name='dropdown' />
                  {head}
                </Accordion.Title>
                <Accordion.Content active={activeHead === head}>
                  <Table celled structured>
                    <Table.Header>
                      <Table.Row>
                        <Table.HeaderCell>Relation</Table.HeaderCell>
                        <Table.HeaderCell>Tails</Table.HeaderCell>
                      </Table.Row>
                    </Table.Header>
                    <Table.Body>
                      {_.map(headResults, (headResult, hrIndex) => {
                        return (
                          <React.Fragment>
                            <Table.Row key={hrIndex}>
                              <Table.Cell rowSpan={headResult['tails'].length > 0 ? headResult['tails'].length : 1} width={3}>{headResult['relation']}</Table.Cell>
                              {headResult['tails'].length > 0 ? <Table.Cell>{_.head(headResult['tails'])}</Table.Cell> : null}
                            </Table.Row>
                            {_.map(_.slice(headResult['tails'], 1), (tail, tIndex) => {
                                return (
                                  <Table.Row key={tIndex}>
                                    <Table.Cell key={tIndex}>{tail}</Table.Cell>
                                  </Table.Row>
                                )
                            })}
                          </React.Fragment>
                        )
                      })}
                    </Table.Body>
                  </Table>
                </Accordion.Content>
              </Accordion>
            </Grid.Column>
          </Grid.Row>
        </Grid>
      )
    })
  }

  const resultPanes = [
    {key: 'Table', menuItem: 'Table', render: () => resultTablePane()},
    {key:'JSON', menuItem: 'JSON', render: () => resultJSONPane}
  ]

  const textInput = (
    <Form>
      <div className='cntr-label'>
        <Popup content='Main text input to extract heads from if enabled, otherwise used as is for knowledge generation' trigger={<Label color='teal'>Text</Label>}/>
      </div>
      <TextArea 
        placeholder='PersonX becomes a great basketball player'
        onChange={e => setText(e.target.value)}
        value={text}
        label='Text'
        rows={2}
      />
    </Form>
  )

  const extractHeadsInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='If enabled, knowledge heads will be extracted from given text using the head processors defined below' trigger={<Label color='teal'>Extract Heads</Label>}/>
      </div>
      <Radio toggle checked={extractHeads} onChange={(e, data) => setExtractHeads(data.checked)}/>
    </React.Fragment>
  )

  const matchRelationsInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='If enabled, (subset of) relations will be matched with extracted heads using the relation processors defined below, otherwise, heads will be matched to all relations given below' trigger={<Label color='teal'>Match Relations</Label>}/>
      </div>
      <Radio toggle checked={matchRelations} onChange={(e, data) => setMatchRelations(data.checked)}/>
    </React.Fragment>
  )

  const dryRunInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content="If enabled, actual knowledge generation through a model won't be run and final input graph to the model will be returned as a result" trigger={<Label color='teal'>Dry Run</Label>}/>
      </div>
      <Radio toggle checked={dryRun} onChange={(e, data) => setDryRun(data.checked)}/>
    </React.Fragment>
  )

  const inferenceFilteringInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content="If enabled, custom context can be provided which will be used to filter out irrelevant generations based on the relevancy threshold" trigger={<Label color='teal'>Inference Filtering</Label>}/>
      </div>
      <Radio toggle checked={inferenceFiltering} onChange={(e, data) => setInferenceFiltering(data.checked)}/>
    </React.Fragment>
  )

  const modelInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='Model to use for knowledge generation' trigger={<Label color='teal'>Model</Label>}/>
      </div>
      <Dropdown
        placeholder='Select Model'
        selection
        options={modelOptions}
        value={model}
        onChange={(e, data) => setModel(data.value)}
      />
    </React.Fragment>
  )

  const contextInput = (
    <Form>
      <div className='cntr-label'>
        <Popup content='Custom context to use for filtering out irrelevant knowledge generations. By default, given text will be used as context' trigger={<Label color='teal'>Context</Label>}/>
      </div>
      <TextArea 
        placeholder={text || 'PersonX becomes a great basketball player'}
        onChange={e => setContext(e.target.value)}
        value={context}
        label='Context'
        rows={3}
      />
    </Form>
  )

  const thresholdInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='Threshold value used for filtering out irrelevant knowledge generations' trigger={<Label color='teal'>Filtering threshold</Label>}/>
      </div>
      <NumberInput
        value={threshold.toString()}
        buttonPlacement="right"
        minValue={0}
        maxValue={1}
        stepAmount={0.1}
        valueType="decimal"
        onChange={setThreshold}
      />
    </React.Fragment>
  )

  const headProcInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='Strategies to use for extracting heads from given text if any' trigger={<Label color='teal'>Head Processors</Label>}/>
      </div>
      <Dropdown
        placeholder='Add Processor'
        selection
        search
        multiple
        options={headProcOptions}
        value={headProcs || []}
        onChange={(e, data) => setHeadProcs(data.value)}
      />
    </React.Fragment>
  )

  const relProcInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='Strategy to use for matching relations with extracted heads if any' trigger={<Label color='teal'>Relation Processors</Label>}/>
      </div>
      <Dropdown
        placeholder='Select Processor'
        selection
        search
        options={relProcOptions}
        value={_.first(relProcs) || ''}
        onChange={(e, data) => setRelProcs([data.value])}
      />
    </React.Fragment>
  )

  const headInput = (
    <Form>
      <div className='cntr-label'>
        <Popup content='Custom head inputs that will be processed as is' trigger={<Label color='teal'>Heads</Label>}/>
      </div>
      <Button icon basic labelPosition='left' onClick={addHead}>
        <Icon name='plus' />
        Add Head
      </Button>
      {getHeadsJSX()}
    </Form>
  )

  const relationInput = (
    <React.Fragment>
      <div className='cntr-label'>
        <Popup content='Subset of relations to match from. By default, all relations are eligible to be matched' trigger={<Label color='teal'>Relations</Label>}/>
      </div>
      <Dropdown
        placeholder='All'
        selection
        search
        multiple
        options={relationOptions}
        value={relations || []}
        onChange={(e, data) => setRelations(data.value)}
      />
    </React.Fragment>
  )

  return (
    <Grid celled="internally" columns={2}>

      <Grid.Column largeScreen={8} computer={10} mobile={16}>
        <Grid container>
          <Grid.Row>
            <Grid.Column width={12}>
              <p className='logo'><span className='logo-k'>K</span>ogito</p>
              <Label.Group>
                <Label as ='a' href='https://arxiv.org/abs/2211.08451' target='_blank' color='blue'>
                  Paper &nbsp;
                  <Icon name='external'/>
                </Label>
                <Label as ='a' href='https://github.com/epfl-nlp/kogito' target='_blank' color='red'>
                  Code &nbsp;
                  <Icon name='external'/>
                </Label>
                <Label as ='a' href='https://kogito.readthedocs.io' target='_blank' color='green'>
                  Docs &nbsp;
                  <Icon name='external'/>
                </Label>
              </Label.Group>
            </Grid.Column>
            <Grid.Column width={4}>
              <Image as='a' src='/lab-logo.png' href='https://nlp.epfl.ch' target='_blank'/>
            </Grid.Column>
          </Grid.Row>
          
          <Grid.Row>
            <Grid.Column>
              <Message color='grey'>
                <Message.Header>A Commonsense Knowledge Inference Toolkit</Message.Header>
                <p className='description'>This is an interactive playground for <b>kogito</b>, a Python library that provides an intuitive interface to generate commonsense knowledge from text. 
                This app is meant to be used for demo purposes only and hence, does not support all available features of the library. Please, hover over the labels to get more information on what each input does and refer to the docs or the paper for additional details.
                </p>
              </Message>
            </Grid.Column>
          </Grid.Row>

          <Grid.Row>
            <Grid.Column>
              <div className='cntr'>
                {textInput}
              </div>
              <div className='cntr'>
                <Grid columns={4}>
                  <Grid.Column computer={4} widescreen={3} tablet={5} mobile={8}>
                    {extractHeadsInput}
                  </Grid.Column>
                  <Grid.Column computer={4} widescreen={3} tablet={5} mobile={8}>
                    {matchRelationsInput}
                  </Grid.Column>
                  <Grid.Column computer={4} widescreen={3} tablet={5} mobile={8}>
                    {dryRunInput}
                  </Grid.Column>
                  <Grid.Column computer={4} widescreen={3} tablet={5} mobile={8}>
                    {inferenceFilteringInput}
                  </Grid.Column>
                  <Grid.Column computer={6} widescreen={4} tablet={10} mobile={16}>
                    {modelInput}
                  </Grid.Column>
                </Grid>
              </div>
              {inferenceFiltering ? 
                <div className='cntr'>
                  <Grid columns={2}>
                    <Grid.Column largeScreen={10} computer={8} tablet={10} mobile={16}>
                      {contextInput}
                    </Grid.Column>
                    <Grid.Column largeScreen={6} computer={8} tablet={6} mobile={16}>
                      {thresholdInput}
                    </Grid.Column>
                  </Grid>
                </div> : null
              }
              <div className='cntr'>
                <Grid columns={2}>
                  {extractHeads ? 
                      <Grid.Column mobile={16} computer={8} tablet={10} largeScreen={10}>
                        {headProcInput}
                      </Grid.Column> : null
                  }
                  {matchRelations ? 
                      <Grid.Column mobile={16} computer={8} tablet={6} largeScreen={6}>
                        {relProcInput}
                      </Grid.Column> : null
                  }
                </Grid>
              </div>
              <div className='cntr'>
                <Grid columns={2}>
                  <Grid.Column mobile={16} computer={10} tablet={10} widescreen={8}>
                    {headInput}
                  </Grid.Column>
                  <Grid.Column mobile={16} computer={6} tablet={6} widescreen={8}>
                    {relationInput}
                  </Grid.Column>
                </Grid>
              </div>
            </Grid.Column>
          </Grid.Row>
        </Grid>
      </Grid.Column>

      <Grid.Column largeScreen={8} computer={6} mobile={16}>
        <Container className='cntr-label'>
          <Label color='black'>Results</Label>
          <Message warning hidden={!resultWarningVisible} onDismiss={() => setResultWarningVisible(false)}>
            <Message.Header>Warning</Message.Header>
            <p className='description'>Please note that this tool might produce a biased or toxic output which can sometimes be mitigated using inference filtering. </p>
          </Message>
        </Container>
        <Container className='cntr'>
          <Button
            onClick={generate}
            className='kbtn'
            loading={generating}
            disabled={generating || (_.isEmpty(_.trim(text)) && (_.isEmpty(heads) || _.every(heads, h => _.isEmpty(_.trim(h)))))}
          >
            Generate
          </Button>
        </Container>
        <Container className='cntr'>
          <Tab menu={{ secondary: true }} panes={resultPanes}/>
        </Container>
        {showError ? 
          <Container>
            <Message
              negative
              header='Error'
              content={errorMsg}
              onDismiss={() => setShowError(false)}
            />
          </Container> : null
        }
        {showNote ?
          <Container>
            <Message
              info
              header='Note'
              content="Inference with GPT-2 models might take longer due to the size of the model and limited capacity of our resources deployed for this demo."
              onDismiss={() => setShowNote(false)}
            />
          </Container> : null

        }
      </Grid.Column>

    </Grid>
  )
}
export default App
